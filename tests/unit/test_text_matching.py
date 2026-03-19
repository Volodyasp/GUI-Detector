from __future__ import annotations

from gui_detector_api.services.text_matching import TextMatchResult, match_text_against_exemplars


def test_exact_match_scores_high():
    result = match_text_against_exemplars(
        "add to cart",
        [("c1", "CartButton", "add to cart")],
    )
    assert result is not None
    assert result.class_name == "CartButton"
    assert result.score >= 0.99


def test_case_insensitive_match():
    result = match_text_against_exemplars(
        "Add To Cart",
        [("c1", "CartButton", "add to cart")],
    )
    assert result is not None
    assert result.score >= 0.99


def test_partial_match_above_threshold():
    result = match_text_against_exemplars(
        "add to cart now",
        [("c1", "CartButton", "add to cart")],
        threshold=0.7,
    )
    assert result is not None
    assert result.class_name == "CartButton"


def test_no_match_below_threshold():
    result = match_text_against_exemplars(
        "product description text here",
        [("c1", "CartButton", "add to cart")],
        threshold=0.65,
    )
    assert result is None


def test_best_match_wins_among_multiple_classes():
    exemplars = [
        ("c1", "CartButton", "add to cart"),
        ("c2", "BuyButton", "buy now"),
    ]
    result = match_text_against_exemplars("buy now", exemplars)
    assert result is not None
    assert result.class_name == "BuyButton"
    assert result.class_id == "c2"


def test_empty_ocr_text_returns_none():
    result = match_text_against_exemplars(
        "",
        [("c1", "CartButton", "add to cart")],
    )
    assert result is None


def test_whitespace_only_ocr_text_returns_none():
    result = match_text_against_exemplars(
        "   ",
        [("c1", "CartButton", "add to cart")],
    )
    assert result is None


def test_empty_exemplars_returns_none():
    result = match_text_against_exemplars("add to cart", [])
    assert result is None


def test_fuzzy_handles_ocr_artifacts():
    result = match_text_against_exemplars(
        "addto cart",
        [("c1", "CartButton", "add to cart")],
        threshold=0.6,
    )
    assert result is not None
    assert result.class_name == "CartButton"


def test_threshold_boundary():
    result = match_text_against_exemplars(
        "completely different text",
        [("c1", "CartButton", "add to cart")],
        threshold=0.9,
    )
    assert result is None


def test_multiple_exemplars_same_class():
    exemplars = [
        ("c1", "CartButton", "add to cart"),
        ("c1", "CartButton", "add to basket"),
    ]
    result = match_text_against_exemplars("add to cart", exemplars)
    assert result is not None
    assert result.class_name == "CartButton"
    assert result.matched_exemplar == "add to cart"
