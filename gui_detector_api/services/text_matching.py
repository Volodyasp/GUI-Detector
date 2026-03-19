from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextMatchResult:
    class_id: str
    class_name: str
    matched_exemplar: str
    score: float


def match_text_against_exemplars(
    ocr_text: str,
    exemplars: list[tuple[str, str, str]],
    *,
    threshold: float = 0.65,
) -> TextMatchResult | None:
    """Match OCR text against class text exemplars using fuzzy matching.

    Args:
        ocr_text: Text extracted by OCR from a crop.
        exemplars: List of (class_id, class_name, text_exemplar) tuples.
        threshold: Minimum fuzzy match score (0.0 - 1.0) to accept.

    Returns:
        Best match above threshold, or None.
    """
    if not ocr_text or not exemplars:
        return None

    normalized_ocr = ocr_text.strip().lower()
    if not normalized_ocr:
        return None

    best: TextMatchResult | None = None
    best_score = 0.0

    for class_id, class_name, exemplar_text in exemplars:
        normalized_exemplar = exemplar_text.strip().lower()
        if not normalized_exemplar:
            continue

        score = _fuzzy_score(normalized_ocr, normalized_exemplar)

        if score > best_score and score >= threshold:
            best_score = score
            best = TextMatchResult(
                class_id=class_id,
                class_name=class_name,
                matched_exemplar=exemplar_text,
                score=score,
            )

    return best


def _fuzzy_score(text_a: str, text_b: str) -> float:
    """Compute fuzzy similarity. Uses rapidfuzz if available, falls back to difflib."""
    try:
        from rapidfuzz import fuzz

        token_sort = fuzz.token_sort_ratio(text_a, text_b) / 100.0
        partial = fuzz.partial_ratio(text_a, text_b) / 100.0
        return max(token_sort, partial)
    except ImportError:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, text_a, text_b).ratio()
