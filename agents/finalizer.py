import structlog
from config import settings
from pipeline.models import Track
from utils.embedding_utils import get_insightface

logger = structlog.get_logger(__name__)


def retry_embedding(track: Track, set_embedding):
    """If track.embedding is None, attempt one last InsightFace detection.

    Uses best_face_crop first, falls back to best_full_frame.
    Accepts a set_embedding callable with signature:
        (composite_id, embedding, is_masked=False, det_score=0.0) -> (bool, reason)
    """
    if track.embedding is not None:
        return

    app = get_insightface()

    crop_faces = []
    if track.best_face_crop is not None:
        try:
            crop_faces = app.detect_faces_raw(track.best_face_crop, min_score=settings.DET_SCORE_RELAXED)
        except Exception as e:
            logger.debug("best_face_crop_detect_failed", track_id=track.track_id, error=str(e))

    frame_faces = []
    if not crop_faces and track.best_full_frame is not None:
        try:
            frame_faces = app.detect_faces_raw(track.best_full_frame, min_score=settings.DET_SCORE_RELAXED)
        except Exception as e:
            logger.debug("best_full_frame_detect_failed", track_id=track.track_id, error=str(e))

    best = None
    source = None
    for faces, src in [(crop_faces, "crop"), (frame_faces, "frame")]:
        if not faces:
            continue
        hit = next((f for f in faces if f["det_score"] >= settings.EMBEDDING_DET_SCORE_MIN), None)
        if hit:
            best, source = hit, src
            break
        if faces[0]["det_score"] >= settings.DET_SCORE_RELAXED:
            best, source = faces[0], f"relaxed_{src}"
            break

    if best:
        embedding = best["embedding"].tolist()
        is_masked = best["is_masked"]
        det_score = best["det_score"]

        accepted, reason = set_embedding(track.track_id, embedding, is_masked=is_masked, det_score=det_score)
        if accepted:
            logger.debug("final_embed_done", track_id=track.track_id, source=source, score=det_score)
        elif reason == "track_removed":
            with track._lock:
                if track.embedding is None or det_score > track.embedding_det_score + 0.05:
                    track.embedding = embedding
                    track.is_masked = is_masked
                    track.embedding_det_score = det_score
                    logger.debug("final_embed_direct",
                                 track_id=track.track_id, source=source, score=det_score,
                                 reason="track_removed")
                else:
                    logger.debug("final_embed_skipped",
                                 track_id=track.track_id,
                                 current_det_score=round(track.embedding_det_score, 3),
                                 new_det_score=round(det_score, 3),
                                 reason="rejected_quality")
        else:
            logger.debug("final_embed_skipped",
                         track_id=track.track_id,
                         current_det_score=round(track.embedding_det_score, 3),
                         new_det_score=round(det_score, 3),
                         reason=reason)
    else:
        logger.info("no_embedding_after_retries",
                    track_id=track.track_id,
                    visibility=track.visibility,
                    frames_seen=track.total_frames_seen,
                    face_detected=track.face_detected_once)