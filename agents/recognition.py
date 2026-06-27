"""
Phase 2.1 + 2.2 — Recognition Agent with Memory Context

Instead of a single if/else on similarity, this agent considers
multiple factors to make a structured decision.

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
from pipeline.models import RecognitionResult
from config import settings

logger = structlog.get_logger(__name__)


class RecognitionAgent(BaseAgent):
    """Makes structured recognition decisions based on multiple signals.

    This agent replaces the simple:
        if similarity > 0.85: VERIFIED
        else: UNKNOWN

    With a richer decision that considers:
    - Similarity score
    - Face quality
    - Mask status
    - Track duration (how long person was visible)
    - Memory context (visit history, patterns)
    """

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute recognition decision.

        Args:
            input_data: Must contain:
                - similarity (float): Cosine similarity from ArcFace
                - is_masked (bool): Whether face is masked
                - face_quality (float): Quality score 0-1
                - track_duration (float): Seconds person was tracked
                - memory_context (dict, optional): From Memory Agent

        Returns:
            Dict with status, confidence, reason, and other fields.
        """
        similarity = input_data.get("similarity", 0.0)
        is_masked = input_data.get("is_masked", False)
        face_quality = input_data.get("face_quality", 0.0)
        track_duration = input_data.get("track_duration", 0.0)
        memory_context = input_data.get("memory_context", {})

        result = self._decide(similarity, is_masked, face_quality, track_duration, memory_context)

        logger.info("recognition_decision",
                    status=result.status,
                    confidence=result.confidence,
                    similarity=result.similarity,
                    masked=result.is_masked,
                    memory_boost=memory_context.get("confidence_boost", 0),
                    reason=result.reason)

        return result.to_dict()

    def _decide(self, similarity: float, is_masked: bool,
                face_quality: float, track_duration: float,
                memory_context: Optional[Dict] = None) -> RecognitionResult:
        """Core decision logic with memory context."""

        # Get memory boost (default 0 if no memory context)
        memory_boost = 0
        visit_count = 0
        if memory_context:
            memory_boost = memory_context.get("confidence_boost", 0)
            visit_count = memory_context.get("visit_count", 0)

        # Case 1: Very high similarity — definitely known
        if similarity >= settings.VERY_HIGH_SIMILARITY:
            confidence = min(95, 70 + (similarity - settings.VERY_HIGH_SIMILARITY) * 250 + memory_boost)
            return RecognitionResult(
                status="known",
                confidence=confidence,
                similarity=similarity,
                face_quality=face_quality,
                is_masked=is_masked,
                track_duration=track_duration,
                reason=self._build_reason("Very high similarity match", similarity, face_quality, is_masked, memory_context)
            )

        # Case 2: Good similarity — likely known
        if similarity >= settings.MATCH_THRESHOLD:
            confidence = self._compute_confidence(similarity, face_quality, track_duration, is_masked, memory_boost)

            # If confidence is high enough, mark as known
            if confidence >= 70:
                status = "known"
                reason = self._build_reason("Good similarity match", similarity, face_quality, is_masked, memory_context)
            else:
                status = "uncertain"
                reason = self._build_reason("Borderline match — needs verification", similarity, face_quality, is_masked, memory_context)

            return RecognitionResult(
                status=status,
                confidence=confidence,
                similarity=similarity,
                face_quality=face_quality,
                is_masked=is_masked,
                track_duration=track_duration,
                reason=reason
            )

        # Case 3: Below threshold but face quality is high — uncertain
        if face_quality >= settings.BORDERLINE_FACE_QUALITY and similarity >= settings.MATCH_THRESHOLD * 0.8:
            return RecognitionResult(
                status="uncertain",
                confidence=40 + similarity * 30 + memory_boost,
                similarity=similarity,
                face_quality=face_quality,
                is_masked=is_masked,
                track_duration=track_duration,
                reason=self._build_reason("Below threshold but good face quality", similarity, face_quality, is_masked, memory_context)
            )

        # Case 4: Low similarity — unknown
        return RecognitionResult(
            status="unknown",
            confidence=max(60, 100 - similarity * 100),
            similarity=similarity,
            face_quality=face_quality,
            is_masked=is_masked,
            track_duration=track_duration,
            reason=self._build_reason("No match found", similarity, face_quality, is_masked, memory_context)
        )

    def _compute_confidence(self, similarity: float, face_quality: float,
                           track_duration: float, is_masked: bool,
                           memory_boost: float = 0) -> float:
        """Compute confidence score (0-100) from multiple signals.

        Higher similarity = higher confidence
        Higher face quality = higher confidence
        Longer track duration = higher confidence (more frames to verify)
        Masked = slightly lower confidence (harder to verify)
        Memory boost = additional confidence from visit history
        """
        # Base confidence from similarity (0-60 points)
        sim_score = min(60, (similarity - settings.MATCH_THRESHOLD) / (1.0 - settings.MATCH_THRESHOLD) * 60)

        # Face quality bonus (0-25 points)
        quality_score = face_quality * 25

        # Track duration bonus (0-15 points)
        # More frames = more chances to verify
        duration_score = min(15, track_duration / 10)

        confidence = sim_score + quality_score + duration_score + memory_boost

        # Mask penalty
        if is_masked:
            confidence *= settings.MASK_CONFIDENCE_PENALITY

        return max(0, min(100, confidence))

    def _build_reason(self, base: str, similarity: float,
                     face_quality: float, is_masked: bool,
                     memory_context: Optional[Dict] = None) -> str:
        """Build a human-readable reason string."""
        parts = [base]

        if similarity >= settings.VERY_HIGH_SIMILARITY:
            parts.append(f"similarity={similarity:.2%}")
        elif similarity >= settings.MATCH_THRESHOLD:
            parts.append(f"similarity={similarity:.2%} above threshold")
        else:
            parts.append(f"similarity={similarity:.2%} below threshold")

        if face_quality >= settings.BORDERLINE_FACE_QUALITY:
            parts.append("high face quality")
        elif face_quality >= 0.5:
            parts.append("medium face quality")
        else:
            parts.append("low face quality")

        if is_masked:
            parts.append("person is masked")

        # Add memory context if available
        if memory_context:
            visit_count = memory_context.get("visit_count", 0)
            if visit_count > 0:
                parts.append(f"returning visitor ({visit_count} previous visits)")

            if memory_context.get("is_typical_time"):
                parts.append("at a typical visit time")

        return ". ".join(parts) + "."


# Convenience function for backward compatibility
def recognize(similarity: float, is_masked: bool = False,
              face_quality: float = 0.0, track_duration: float = 0.0,
              memory_context: Optional[Dict] = None) -> dict:
    """Quick recognition without instantiating the agent."""
    agent = RecognitionAgent()
    return agent.run({
        "similarity": similarity,
        "is_masked": is_masked,
        "face_quality": face_quality,
        "track_duration": track_duration,
        "memory_context": memory_context or {},
    })
