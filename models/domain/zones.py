"""Shared region partitioning and per-zone detection selection."""


class ZoneLayout:
    """Map detection centers to configured bean and digit zones.

    The layout retains the supplied ``lines`` mapping by reference so calibration
    adjustments are immediately reflected by both rendering and task generation.
    """

    _BEAN_KEYS = ("left_x", "right_x")
    _DIGIT_KEYS = ("x1", "x2", "x3", "x4")

    def __init__(self, lines):
        self.lines = lines

    @staticmethod
    def _is_number(value):
        return isinstance(value, (int, float))

    def validate(self):
        """Return a configuration error string, or ``None`` when valid."""
        if not isinstance(self.lines, dict):
            return "invalid lines config"
        bean = self.lines.get("bean")
        digit = self.lines.get("digit")
        if not isinstance(bean, dict) or not all(k in bean for k in self._BEAN_KEYS):
            return "invalid bean line config"
        if not isinstance(digit, dict) or not all(k in digit for k in self._DIGIT_KEYS):
            return "invalid digit line config"
        if not all(self._is_number(bean[k]) for k in self._BEAN_KEYS):
            return "invalid bean line coordinate"
        if not all(self._is_number(digit[k]) for k in self._DIGIT_KEYS):
            return "invalid digit line coordinate"
        if any(bean[k] < 0 for k in self._BEAN_KEYS):
            return "bean line coordinates must be non-negative"
        if any(digit[k] < 0 for k in self._DIGIT_KEYS):
            return "digit line coordinates must be non-negative"
        if bean["left_x"] >= bean["right_x"]:
            return "bean left_x must be less than right_x"
        xs = [digit[k] for k in self._DIGIT_KEYS]
        if not all(xs[i] < xs[i + 1] for i in range(len(xs) - 1)):
            return "digit x1..x4 must be strictly increasing"
        return None

    def bean_zone(self, cx):
        """Return pickup zone 1-3; a boundary point belongs to the right zone."""
        left = self.lines["bean"]["left_x"]
        right = self.lines["bean"]["right_x"]
        if cx < left:
            return 1
        if cx < right:
            return 2
        return 3

    def digit_zone(self, cx, _cy=None):
        """Return placement zone 4-8; rows share the same x partitions."""
        xs = [self.lines["digit"][key] for key in self._DIGIT_KEYS]
        if cx < xs[0]:
            return 4
        if cx < xs[1]:
            return 5
        if cx < xs[2]:
            return 6
        if cx < xs[3]:
            return 7
        return 8

    def zone_for(self, mode, cx, cy=None):
        if mode == "bean":
            return self.bean_zone(cx)
        if mode == "digit":
            return self.digit_zone(cx, cy)
        raise ValueError(f"unknown detection mode: {mode!r}")

    @staticmethod
    def zones_for(mode):
        if mode == "bean":
            return (1, 2, 3)
        if mode == "digit":
            return (4, 5, 6, 7, 8)
        raise ValueError(f"unknown detection mode: {mode!r}")

    @staticmethod
    def detection_center(det):
        if not isinstance(det, (list, tuple)) or len(det) < 6:
            raise ValueError("malformed detection")
        try:
            x1, y1, x2, y2, confidence = det[:5]
            return (x1 + x2) / 2, (y1 + y2) / 2, float(confidence)
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("malformed detection") from exc

    def select_best(self, dets, mode, *, ignore_invalid=False):
        """Return ``{zone: detection | None}``, keeping only the best detection.

        Task generation uses strict validation, while rendering ignores malformed
        third-party detections so one bad box cannot interrupt the UI refresh.
        """
        best = {zone: None for zone in self.zones_for(mode)}
        for det in dets:
            try:
                cx, cy, confidence = self.detection_center(det)
            except ValueError:
                if ignore_invalid:
                    continue
                raise
            zone = self.zone_for(mode, cx, cy)
            current = best[zone]
            if current is None or confidence > float(current[4]):
                best[zone] = det
        return best
