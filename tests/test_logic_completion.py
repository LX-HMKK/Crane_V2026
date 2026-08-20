#!/usr/bin/env python3
"""Standalone test for the logical completion module.

Exercises validate_and_complete() and generate_frame_with_completion()
with hand-crafted detection inputs covering all major edge cases.
No cameras, models, or serial port required.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.domain.logic_completion import count_inferred, validate_and_complete
from models.domain.mapping import DEFAULT_BEAN_TO_DIGIT
from models.domain.task_generator import TaskGenerator

# ── helpers ────────────────────────────────────────────────────────────────

LINES = {
    "bean": {"left_x": 500, "right_x": 800},
    "digit": {"x1": 200, "x2": 400, "x3": 600, "x4": 800},
}
BEAN_NAMES = {0: "yellow", 1: "green", 2: "white"}
DIGIT_NAMES = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
MAPPING = DEFAULT_BEAN_TO_DIGIT

TG = TaskGenerator(MAPPING, LINES, BEAN_NAMES, DIGIT_NAMES)


def _bean_det(cls_id, cx, cy=360, conf=0.9):
    """Shorthand: [x1,y1,x2,y2,conf,cls_id] with 60x60 box around cx,cy."""
    return [cx - 30, cy - 30, cx + 30, cy + 30, float(conf), cls_id]


def _digit_det(cls_id, cx, cy=360, conf=0.9):
    return _bean_det(cls_id, cx, cy, conf)


def _source_summary(results):
    """Human-readable per-zone source string."""
    parts = []
    for z in sorted(results):
        val, src, conf = results[z]
        c = f"{conf:.2f}" if conf is not None else "-"
        parts.append(f"Z{z}={val}({src},{c})")
    return " ".join(parts)


# ── test cases ─────────────────────────────────────────────────────────────

def test_all_detected():
    """All zones have unique, high-confidence detections."""
    bean_dets = [
        _bean_det(0, 300),   # yellow in zone 1
        _bean_det(1, 600),   # green  in zone 2
        _bean_det(2, 900),   # white  in zone 3
    ]
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        _digit_det(2, 500), _digit_det(3, 700),
        _digit_det(4, 900),
    ]
    payload, br, dr, reason = TG.generate_frame_with_completion(
        bean_dets, digit_dets,
    )
    assert payload == (1, 2, 3, 1, 2, 3, 4, 5), f"got {payload}"
    assert count_inferred(br) == 0
    assert count_inferred(dr) == 0
    print("PASS test_all_detected")


def test_one_bean_missing():
    """Zone 2 has no detection → color 2 inferred to zone 2."""
    bean_dets = [
        _bean_det(0, 300, conf=0.9),   # yellow Z1
        # Z2: empty
        _bean_det(2, 900, conf=0.8),   # white Z3
    ]
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        _digit_det(2, 500), _digit_det(3, 700), _digit_det(4, 900),
    ]
    payload, br, dr, reason = TG.generate_frame_with_completion(
        bean_dets, digit_dets,
    )
    assert payload is not None, f"reason={reason}"
    assert payload[:3] == (1, 2, 3), f"got C1-C3={payload[:3]}"
    assert br[2][1] == "inferred", f"Z2 should be inferred, got {br[2]}"
    assert br[1][1] == "detected"
    print("PASS test_one_bean_missing")


def test_one_digit_missing():
    """Zone 6 empty → digit 3 inferred."""
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        # Z6: empty
        _digit_det(3, 700), _digit_det(4, 900),
    ]
    bean_dets = [
        _bean_det(0, 300), _bean_det(1, 600), _bean_det(2, 900),
    ]
    payload, br, dr, reason = TG.generate_frame_with_completion(
        bean_dets, digit_dets,
    )
    assert payload is not None, f"reason={reason}"
    assert payload[3:] == (1, 2, 3, 4, 5), f"got N4-N8={payload[3:]}"
    assert dr[6][1] == "inferred", f"Z6 should be inferred, got {dr[6]}"
    print("PASS test_one_digit_missing")


def test_bean_duplicate_resolved():
    """Z1 detected yellow(0.55), Z2 misdetected as yellow(0.90).
    Permutation scoring should assign Z1=2(inferred), Z2=1(detected)."""
    bean_dets = [
        _bean_det(0, 300, conf=0.55),  # yellow(1) in Z1, low conf
        _bean_det(0, 600, conf=0.90),  # yellow(1) in Z2, misclassified
        _bean_det(2, 900, conf=0.80),  # white(3) in Z3
    ]
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        _digit_det(2, 500), _digit_det(3, 700), _digit_det(4, 900),
    ]
    payload, br, dr, reason = TG.generate_frame_with_completion(
        bean_dets, digit_dets,
    )
    assert payload is not None, f"reason={reason}"
    # Best permutation: Z1=2(0)+Z2=1(0.90)+Z3=3(0.80)=1.70
    # vs Z1=1(0.55)+Z2=2(0)+Z3=3(0.80)=1.35
    assert br[2][0] == 1 and br[2][1] == "detected", \
        f"Z2 should get value 1 detected, got {br[2]}"
    assert br[1][0] == 2 and br[1][1] == "inferred", \
        f"Z1 should get value 2 inferred, got {br[1]}"
    print("PASS test_bean_duplicate_resolved")


def test_too_many_bean_missing():
    """2 of 3 bean zones empty → exceeds max_inferred_beans=1."""
    bean_dets = [
        _bean_det(0, 300, conf=0.9),
        # Z2, Z3: empty
    ]
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        _digit_det(2, 500), _digit_det(3, 700), _digit_det(4, 900),
    ]
    payload, br, dr, reason = TG.generate_frame_with_completion(
        bean_dets, digit_dets,
    )
    assert payload is None, f"should be None, got {payload}"
    assert "inferred bean" in reason, f"reason={reason}"
    print("PASS test_too_many_bean_missing")


def test_too_many_digit_missing():
    """3 of 5 digit zones empty → exceeds max_inferred_digits=2."""
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        # Z6, Z7, Z8: empty
    ]
    bean_dets = [
        _bean_det(0, 300), _bean_det(1, 600), _bean_det(2, 900),
    ]
    payload, br, dr, reason = TG.generate_frame_with_completion(
        bean_dets, digit_dets,
    )
    assert payload is None, f"should be None, got {payload}"
    assert "inferred digit" in reason, f"reason={reason}"
    print("PASS test_too_many_digit_missing")


def test_unknown_bean_class_ignored():
    """cls_id=99 (unknown) in Z1 → treated as empty, inferred to 1."""
    bean_dets = [
        _bean_det(99, 300, conf=0.9),  # unknown class
        _bean_det(1, 600, conf=0.9),   # green(2) in Z2
        _bean_det(2, 900, conf=0.9),   # white(3) in Z3
    ]
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        _digit_det(2, 500), _digit_det(3, 700), _digit_det(4, 900),
    ]
    payload, br, dr, reason = TG.generate_frame_with_completion(
        bean_dets, digit_dets,
    )
    assert payload is not None, f"reason={reason}"
    assert br[1][0] == 1 and br[1][1] == "inferred", \
        f"Z1 should be inferred 1, got {br[1]}"
    assert count_inferred(br) == 1
    print("PASS test_unknown_bean_class_ignored")


def test_low_confidence_still_scored():
    """Detection at 0.28 confidence is below threshold but still scored.
    Should be detected (it's the only detection for that value)."""
    bean_dets = [
        _bean_det(0, 300, conf=0.90),  # yellow Z1
        _bean_det(1, 600, conf=0.28),  # green Z2, very low conf
        _bean_det(2, 900, conf=0.80),  # white Z3
    ]
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        _digit_det(2, 500), _digit_det(3, 700), _digit_det(4, 900),
    ]
    payload, br, dr, reason = TG.generate_frame_with_completion(
        bean_dets, digit_dets,
    )
    assert payload is not None, f"reason={reason}"
    # Permutation scoring: (1,2,3)=0.90+0.28+0.80=1.98 vs any other < 1.98
    assert br[2][0] == 2 and br[2][1] == "detected", \
        f"Z2 should be detected (only detection for value 2), got {br[2]}"
    assert count_inferred(br) == 0
    print("PASS test_low_confidence_still_scored")


def test_backward_compat_generate_frame():
    """Original generate_frame() still works for valid inputs."""
    bean_dets = [
        _bean_det(0, 300), _bean_det(1, 600), _bean_det(2, 900),
    ]
    digit_dets = [
        _digit_det(0, 100), _digit_det(1, 300),
        _digit_det(2, 500), _digit_det(3, 700), _digit_det(4, 900),
    ]
    payload, reason = TG.generate_frame(bean_dets, digit_dets)
    assert payload == (1, 2, 3, 1, 2, 3, 4, 5)
    assert reason == "ok"
    print("PASS test_backward_compat_generate_frame")


def test_standalone_validate():
    """Directly test validate_and_complete() with simple inputs."""
    # All detected
    r = validate_and_complete(
        {1: (1, 0.9), 2: (2, 0.8), 3: (3, 0.7)},
        {1, 2, 3},
    )
    assert r[1] == (1, "detected", 0.9)
    assert r[2] == (2, "detected", 0.8)
    assert r[3] == (3, "detected", 0.7)
    assert count_inferred(r) == 0

    # One missing
    r = validate_and_complete(
        {1: (1, 0.9), 2: None, 3: (3, 0.7)},
        {1, 2, 3},
    )
    assert r[2] == (2, "inferred", None)
    assert count_inferred(r) == 1

    # Duplicate resolved
    r = validate_and_complete(
        {1: (1, 0.55), 2: (1, 0.90), 3: (3, 0.80)},
        {1, 2, 3},
    )
    assert r[2][0] == 1 and r[2][1] == "detected"
    assert r[1][0] == 2 and r[1][1] == "inferred"

    # 5-digit case: two missing
    r = validate_and_complete(
        {4: (1, 0.9), 5: (2, 0.8), 6: None, 7: None, 8: (5, 0.7)},
        {1, 2, 3, 4, 5},
    )
    assert r[6][0] == 3 and r[6][1] == "inferred"
    assert r[7][0] == 4 and r[7][1] == "inferred"
    assert count_inferred(r) == 2

    print("PASS test_standalone_validate")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=== Logic Completion Standalone Tests ===\n")
    tests = [
        test_all_detected,
        test_one_bean_missing,
        test_one_digit_missing,
        test_bean_duplicate_resolved,
        test_too_many_bean_missing,
        test_too_many_digit_missing,
        test_unknown_bean_class_ignored,
        test_low_confidence_still_scored,
        test_backward_compat_generate_frame,
        test_standalone_validate,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")


if __name__ == "__main__":
    main()
