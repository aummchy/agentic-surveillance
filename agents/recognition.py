"""
Phase 2.1 + 2.2 — Recognition Agent with Memory Context

This module is now orchestration-only. All scoring logic lives in
agents/scoring.py, all tunables live in config/settings.py.

Input:
{
    "similarity": 0.83,
    "is_masked": false,
    "face_quality": 0.91,
    "track_duration": 18,
    "memory_context": {                    # Optional, from Memory Agent
        "visit_count": 5,
        "is_typical_time": true,
        "confidence_boost": 10
    }
}

Output:
{
    "status": "known",
    "confidence": 82,
    "reason": "Returning visitor, seen 5 times before. Borderline similarity but face quality is high."
}
"""

import structlog
from typing import Any, Dict, Optional
from agents.base import BaseAgent
from agents.scoring import compute_confidence, is_match, confidence_status
from pipeline.models import RecognitionResult
from config import settings

logger = structlog.get_logger(__name__)


class RecognitionAgent(BaseAgent):
    """Makes structured recognition decisions based on multiple signals.

    Scoring is delegated to agents.scoring module. This class handles
    orchestration: extract inputs, call scorer, build result.
    """

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recognition decision.

        Args:
            input_data: Must contain:
                - similarity (float): Cosine similarity from ArcFace
                - is_masked (bool): Whether face is masked
                - face_quality (float): Quality score 0-1 (or None/0 for missing)
                - track_duration (float): Seconds person was tracked
                - memory_context (dict, optional): From Memory Agent
                - top2 (float, optional): Second-best similarity
                - margin (float, optional): Margin between top1 and top2
                - name (str, optional): Person name for display

        Returns:
            Dict with status, confidence, reason, and other fields.
        """
        similarity = input_data.get("similarity", 0.0)
        is_masked = input_data.get("is_masked", False)
        face_quality = input_data.get("face_quality", 0.0)
        track_duration = input_data.get("track_duration", 0.0)
        memory_context = input_data.get("memory_context", {})
        top2 = input_data.get("top2")
        margin = input_data.get("margin")
        name = input_data.get("name")

        result = self._decide(similarity, is_masked, face_quality, track_duration, memory_context, margin=margin)

        logger.info("recognition_decision",
                    status=result.status,
                    confidence=result.confidence,
                    similarity=result.similarity,
                    top2=top2,
                    margin=margin,
                    face_quality=face_quality,
                    track_duration=track_duration,
                    masked=result.is_masked,
                    memory_boost=memory_context.get("confidence_boost", 0),
                    visit_count=memory_context.get("visit_count", 0),
                    name=name,
                    reason=result.reason)

        return result.to_dict()

    def _decide(self, similarity: float, is_masked: bool,
                face_quality: float, track_duration: float,
                memory_context: Optional[Dict] = None,
                margin: float = None) -> RecognitionResult:
        """Core decision logic that delegates to the scorer module."""
        memory_boost = 0.0
        if memory_context:
            memory_boost = memory_context.get("confidence_boost", 0.0)

        matched = is_match(similarity)

        confidence = compute_confidence(
            raw_cosine=similarity,
            face_quality=face_quality,
            track_seconds=track_duration,
            memory_boost=memory_boost,
            is_masked=is_masked,
            margin=margin,
        )

        status = confidence_status(confidence, matched)

        return RecognitionResult(
            status=status,
            confidence=confidence,
            similarity=similarity,
            face_quality=face_quality,
            is_masked=is_masked,
            track_duration=track_duration,
            reason=self._build_reason(status, similarity, face_quality, is_masked, memory_context),
        )

    def _build_reason(self, status: str, similarity: float,
                      face_quality: float, is_masked: bool,
                      memory_context: Optional[Dict] = None) -> str:
        """Build a human-readable reason string."""
        parts = [f"Decision: {status}"]

        threshold = settings.MATCH_THRESHOLD
        matched = similarity >= threshold
        if matched:
            parts.append(f"raw cosine={similarity:.3f}, above match threshold {threshold}")
        else:
            parts.append(f"raw cosine={similarity:.3f}, below match threshold {threshold}")

        if face_quality is None or face_quality <= 0.0:
            parts.append("face quality unavailable")
        elif face_quality >= 0.55:
            parts.append("good face quality")
        elif face_quality >= 0.25:
            parts.append("usable face quality")
        else:
            parts.append("low face quality")

        if is_masked:
            parts.append("person is masked")

        if memory_context:
            visit_count = memory_context.get("visit_count", 0)
            if visit_count > 0:
                parts.append(f"returning visitor ({visit_count} previous visits)")
            if memory_context.get("is_typical_time"):
                parts.append("at a typical visit time")

        return ". ".join(parts) + "."


def recognize(similarity: float, is_masked: bool = False,
              face_quality: float = 0.0, track_duration: float = 0.0,
              memory_context: Optional[Dict] = None,
              margin: float = None) -> dict:
    """Quick recognition without instantiating the agent."""
    agent = RecognitionAgent()
    return agent.run({
        "similarity": similarity,
        "is_masked": is_masked,
        "face_quality": face_quality,
        "track_duration": track_duration,
        "memory_context": memory_context or {},
        "margin": margin,
    })
