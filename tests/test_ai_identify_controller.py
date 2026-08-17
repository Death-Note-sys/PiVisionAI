import pytest
import numpy as np
from unittest.mock import MagicMock
from app.modules.ai_identify.controller import AIIdentifyController
from app.modules.ai_identify.settings import AIIdentifySettings
from app.core.models.results import IdentifyResult
import cv2


@pytest.fixture
def controller():
    event_bus = MagicMock()
    settings = AIIdentifySettings()
    return AIIdentifyController(event_bus, settings)


def make_checkerboard(size=200, square=20, invert=False):
    board = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, square):
        for j in range(0, size, square):
            is_white = (i // square + j // square) % 2 == 0
            if invert:
                is_white = not is_white
            if is_white:
                board[i:i+square, j:j+square] = 255
    return board


def make_textured_pattern(size=200, seed=42):
    """Locally-unique synthetic texture for ORB matching tests. Unlike a
    checkerboard, no two regions look alike, so Lowe's ratio test can
    correctly disambiguate real matches — a checkerboard's repeating
    identical corners cause every match to look equally ambiguous, which
    is real, correct ratio-test behavior, not a bug to work around."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (size, size), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (5, 5), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def test_process_before_teaching_returns_untaught_status(controller):
    frame = make_checkerboard()
    result = controller.process({"frame": frame})

    assert isinstance(result, IdentifyResult)
    assert result.teach_status == "Untaught"
    assert result.located is False


def test_process_stores_last_frame_even_when_untaught(controller):
    frame = make_checkerboard()
    controller.process({"frame": frame})

    assert controller.last_frame is not None


def test_teach_good_fails_with_no_frame_yet(controller):
    result = controller.teach_good(10, 10, 50, 50)
    assert result is False


def test_teach_good_rejects_region_too_small(controller):
    frame = make_checkerboard()
    controller.process({"frame": frame})

    result = controller.teach_good(10, 10, 2, 2)
    assert result is False
    assert len(controller.good_references) == 0


def test_teach_good_succeeds_on_feature_rich_region(controller):
    frame = make_checkerboard()
    controller.process({"frame": frame})

    result = controller.teach_good(0, 0, 200, 200)
    assert result is True
    assert len(controller.good_references) == 1
    assert controller.teach_status == "Partial"


def test_teach_status_becomes_taught_after_both_references(controller):
    frame = make_checkerboard()
    controller.process({"frame": frame})
    controller.teach_good(0, 0, 200, 200)
    controller.teach_bad(0, 0, 200, 200)

    assert controller.teach_status == "Taught"


def test_reset_teaching_clears_all_state(controller):
    frame = make_checkerboard()
    controller.process({"frame": frame})
    controller.teach_good(0, 0, 200, 200)
    controller.teach_bad(0, 0, 200, 200)

    result = controller.reset_teaching()

    assert result is True
    assert controller.teach_status == "Untaught"
    assert len(controller.good_references) == 0
    assert len(controller.bad_references) == 0


def test_configure_updates_settings(controller):
    result = controller.configure({"min_match_count": 20})
    assert result is True
    assert controller.settings.get_settings()["min_match_count"] == 20


def test_process_classifies_via_geometric_match_when_only_good_matches(controller):
    """When Bad's content is genuinely unrelated to the live frame (as with
    two different textures — or, in the real bug this fixes, two visually
    distinct physical objects), Bad should fail to find a valid geometric
    match at all, and classification should come from the identity-match
    branch, not an invalid SSIM comparison against unrelated content."""
    good_frame = make_textured_pattern(seed=1)
    controller.process({"frame": good_frame})
    taught = controller.teach_good(0, 0, 200, 200)
    assert taught is True

    bad_frame = make_textured_pattern(seed=2)
    controller.last_frame = bad_frame
    controller.teach_bad(0, 0, 200, 200)
    assert controller.teach_status == "Taught"

    result = controller.process({"frame": good_frame})

    assert result.located is True
    assert result.classification == "Good"
    assert result.good_similarity is not None
    assert result.bad_similarity is None


def test_teach_good_appends_multiple_images(controller):
    frame1 = make_textured_pattern(seed=1)
    controller.process({"frame": frame1})
    controller.teach_good(0, 0, 200, 200)

    frame2 = make_textured_pattern(seed=3)
    controller.last_frame = frame2
    result = controller.teach_good(0, 0, 200, 200)

    assert result is True
    assert len(controller.good_references) == 2


def test_teach_good_rejects_beyond_max_gallery_size(controller):
    for seed in range(10, 15):
        frame = make_textured_pattern(seed=seed)
        controller.last_frame = frame
        controller.teach_good(0, 0, 200, 200)

    assert len(controller.good_references) == 5

    overflow_frame = make_textured_pattern(seed=99)
    controller.last_frame = overflow_frame
    result = controller.teach_good(0, 0, 200, 200)

    assert result is False
    assert len(controller.good_references) == 5


def test_remove_good_reference_by_index(controller):
    frame1 = make_textured_pattern(seed=1)
    controller.process({"frame": frame1})
    controller.teach_good(0, 0, 200, 200)

    frame2 = make_textured_pattern(seed=3)
    controller.last_frame = frame2
    controller.teach_good(0, 0, 200, 200)

    assert len(controller.good_references) == 2

    result = controller.remove_good_reference(0)

    assert result is True
    assert len(controller.good_references) == 1


def test_remove_good_reference_invalid_index_returns_false(controller):
    result = controller.remove_good_reference(0)
    assert result is False


def test_remove_good_reference_updates_teach_status(controller):
    frame1 = make_textured_pattern(seed=1)
    controller.process({"frame": frame1})
    controller.teach_good(0, 0, 200, 200)

    frame2 = make_textured_pattern(seed=2)
    controller.last_frame = frame2
    controller.teach_bad(0, 0, 200, 200)

    assert controller.teach_status == "Taught"

    controller.remove_bad_reference(0)

    assert controller.teach_status == "Partial"


def test_process_reports_reference_counts_in_result(controller):
    good_frame = make_textured_pattern(seed=1)
    controller.process({"frame": good_frame})
    controller.teach_good(0, 0, 200, 200)

    bad_frame = make_textured_pattern(seed=2)
    controller.last_frame = bad_frame
    controller.teach_bad(0, 0, 200, 200)

    result = controller.process({"frame": good_frame})

    assert result.good_reference_count == 1
    assert result.bad_reference_count == 1
