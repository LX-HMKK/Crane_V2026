import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Keep this module independent of the heavy inference dependencies.
sys.modules["ultralytics"] = None
sys.modules["torch"] = None

import pytest
from models.domain.logic_completion import count_inferred, validate_and_complete
from models.domain.mapping import DEFAULT_BEAN_TO_DIGIT
from models.domain.task_generator import TaskGenerator


# ── shared fixtures ────────────────────────────────────────────────────────

LINES = {
    "bean": {"left_x": 500, "right_x": 800},
    "digit": {"x1": 200, "x2": 400, "x3": 600, "x4": 800},
}
BEAN_NAMES = {0: "yellow", 1: "green", 2: "white"}
DIGIT_NAMES = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}


def _bean_det(cls_id, cx, cy=360, conf=0.9):
    return [cx - 30, cy - 30, cx + 30, cy + 30, float(conf), cls_id]


def _digit_det(cls_id, cx, cy=360, conf=0.9):
    return _bean_det(cls_id, cx, cy, conf)


@pytest.fixture
def task_gen():
    return TaskGenerator(DEFAULT_BEAN_TO_DIGIT, LINES, BEAN_NAMES, DIGIT_NAMES)


# ── validate_and_complete standalone tests ──────────────────────────────────


class TestValidateAndComplete:
    def test_all_detected_no_inference(self):
        r = validate_and_complete(
            {1: (1, 0.9), 2: (2, 0.8), 3: (3, 0.7)},
            {1, 2, 3},
        )
        assert r[1] == (1, "detected", 0.9)
        assert r[2] == (2, "detected", 0.8)
        assert r[3] == (3, "detected", 0.7)
        assert count_inferred(r) == 0

    def test_one_missing_inferred(self):
        r = validate_and_complete(
            {1: (1, 0.9), 2: None, 3: (3, 0.7)},
            {1, 2, 3},
        )
        assert r[2] == (2, "inferred", None)
        assert count_inferred(r) == 1

    def test_duplicate_permutation_scoring(self):
        r = validate_and_complete(
            {1: (1, 0.55), 2: (1, 0.90), 3: (3, 0.80)},
            {1, 2, 3},
        )
        # Best perm: Z1=2(0)+Z2=1(0.90)+Z3=3(0.80)=1.70
        assert r[2][0] == 1 and r[2][1] == "detected"
        assert r[1][0] == 2 and r[1][1] == "inferred"

    def test_all_empty_defaults_to_first_permutation(self):
        r = validate_and_complete(
            {1: None, 2: None, 3: None},
            {1, 2, 3},
        )
        assert r[1] == (1, "inferred", None)
        assert r[2] == (2, "inferred", None)
        assert r[3] == (3, "inferred", None)
        assert count_inferred(r) == 3

    def test_two_missing(self):
        r = validate_and_complete(
            {4: (1, 0.9), 5: (2, 0.8), 6: None, 7: None, 8: (5, 0.7)},
            {1, 2, 3, 4, 5},
        )
        assert r[6][0] == 3 and r[6][1] == "inferred"
        assert r[7][0] == 4 and r[7][1] == "inferred"
        assert count_inferred(r) == 2

    def test_out_of_range_value_treated_as_empty(self):
        r = validate_and_complete(
            {1: (1, 0.9), 2: (99, 0.8), 3: (3, 0.7)},
            {1, 2, 3},
        )
        assert r[2] == (2, "inferred", None)
        assert count_inferred(r) == 1

    def test_mismatched_zone_count_raises(self):
        with pytest.raises(ValueError):
            validate_and_complete({1: (1, 0.9), 2: (2, 0.8)}, {1, 2, 3})

    def test_five_zone_digit_case(self):
        r = validate_and_complete(
            {4: (1, 0.9), 5: (3, 0.85), 6: (4, 0.8), 7: (5, 0.75), 8: None},
            {1, 2, 3, 4, 5},
        )
        assert r[8][0] == 2 and r[8][1] == "inferred"
        assert count_inferred(r) == 1

    def test_all_same_value_duplicate(self):
        """All 3 zones claim value 1 with different confidences."""
        r = validate_and_complete(
            {1: (1, 0.9), 2: (1, 0.5), 3: (1, 0.3)},
            {1, 2, 3},
        )
        # Should pick the highest-confidence assignment
        # Best perm assigns 1 to Z1 (0.9) and infers Z2, Z3
        assert r[1][0] == 1 and r[1][1] == "detected"
        assert count_inferred(r) == 2


# ── generate_frame_with_completion integration tests ────────────────────────


class TestGenerateFrameWithCompletion:
    def test_all_detected(self, task_gen):
        bean = [_bean_det(0, 300), _bean_det(1, 600), _bean_det(2, 900)]
        digit = [_digit_det(0, 100), _digit_det(1, 300), _digit_det(2, 500),
                 _digit_det(3, 700), _digit_det(4, 900)]
        p, br, dr, reason = task_gen.generate_frame_with_completion(bean, digit)
        assert p == (1, 2, 3, 1, 2, 3, 4, 5)
        assert reason == "ok"
        assert count_inferred(br) == 0
        assert count_inferred(dr) == 0

    def test_one_bean_missing(self, task_gen):
        bean = [_bean_det(0, 300), _bean_det(2, 900)]
        digit = [_digit_det(0, 100), _digit_det(1, 300), _digit_det(2, 500),
                 _digit_det(3, 700), _digit_det(4, 900)]
        p, br, dr, reason = task_gen.generate_frame_with_completion(bean, digit)
        assert p is not None
        assert br[2][1] == "inferred"
        assert count_inferred(br) == 1

    def test_too_many_bean_missing(self, task_gen):
        bean = [_bean_det(0, 300)]  # only 1 of 3
        digit = [_digit_det(0, 100), _digit_det(1, 300), _digit_det(2, 500),
                 _digit_det(3, 700), _digit_det(4, 900)]
        p, br, dr, reason = task_gen.generate_frame_with_completion(bean, digit)
        assert p is None
        assert "inferred bean" in reason

    def test_unknown_class_ignored(self, task_gen):
        bean = [_bean_det(99, 300, conf=0.9), _bean_det(1, 600), _bean_det(2, 900)]
        digit = [_digit_det(0, 100), _digit_det(1, 300), _digit_det(2, 500),
                 _digit_det(3, 700), _digit_det(4, 900)]
        p, br, dr, reason = task_gen.generate_frame_with_completion(bean, digit)
        assert p is not None
        assert count_inferred(br) == 1  # Z1 inferred
