from models.mapping import bean_name_to_digit


class TaskGenerator:
    """Convert bean/digit detections into task results."""

    _BEAN_KEYS = ("left_x", "right_x")
    _DIGIT_KEYS = ("x1", "x2", "x3", "x4")

    def __init__(self, mapping, lines, bean_names, digit_names):
        self.mapping = mapping
        self.lines = lines
        self.bean_names = bean_names
        self.digit_names = digit_names

    @staticmethod
    def _is_number(value):
        return isinstance(value, (int, float))

    def _validate_lines(self):
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

    @staticmethod
    def _validate_detection(det):
        if not isinstance(det, (list, tuple)) or len(det) < 6:
            return "malformed detection"
        return None

    def _bean_zone(self, cx):
        # Strict < is intentional: detections exactly on a boundary line fall
        # into the zone to the right of that line.
        left = self.lines["bean"]["left_x"]
        right = self.lines["bean"]["right_x"]
        if cx < left:
            return 1
        if cx < right:
            return 2
        return 3

    def _digit_zone(self, cx, cy):
        # 5 placement baskets arranged left-to-right; partition by x only.
        xs = [self.lines["digit"][k] for k in self._DIGIT_KEYS]
        if cx < xs[0]:
            return 4
        if cx < xs[1]:
            return 5
        if cx < xs[2]:
            return 6
        if cx < xs[3]:
            return 7
        return 8

    def _center(self, det):
        x1, y1, x2, y2 = det[:4]
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _bean_color_code(self, name):
        """Map bean class name to protocol color code (1=yellow, 2=green, 3=white)."""
        return bean_name_to_digit(name, self.mapping)

    def _aggregate_beans(self, bean_dets):
        """Return {zone: det} for pickup zones 1-3, keeping highest-confidence detection."""
        best = {1: None, 2: None, 3: None}
        for det in bean_dets:
            err = self._validate_detection(det)
            if err:
                return None, err
            cx, cy = self._center(det)
            zone = self._bean_zone(cx)
            if best[zone] is None or det[4] > best[zone][4]:
                best[zone] = det
        return best, None

    def _aggregate_digits(self, digit_dets):
        """Return {zone: det} for placement zones 4-8, keeping highest-confidence detection."""
        best = {4: None, 5: None, 6: None, 7: None, 8: None}
        for det in digit_dets:
            err = self._validate_detection(det)
            if err:
                return None, err
            cx, cy = self._center(det)
            zone = self._digit_zone(cx, cy)
            if best[zone] is None or det[4] > best[zone][4]:
                best[zone] = det
        return best, None

    def generate(self, bean_dets, digit_dets):
        """Legacy triplet: return (p1, p2, p3) placement zones for targets 1,2,3.

        Kept for backward compatibility with main_amd64.py and existing tests.
        """
        err = self._validate_lines()
        if err:
            return None, err

        for name, value in (
            ("mapping", self.mapping),
            ("bean_names", self.bean_names),
            ("digit_names", self.digit_names),
        ):
            if not isinstance(value, dict):
                return None, f"invalid {name}"

        bean_best, err = self._aggregate_beans(bean_dets)
        if err:
            return None, err

        target_digits = {}
        for zone in (1, 2, 3):
            det = bean_best[zone]
            if det is None:
                return None, f"missing bean in pickup zone {zone}"
            try:
                cls_id = int(det[5])
            except (ValueError, TypeError):
                return None, f"invalid bean class in zone {zone}"
            name = self.bean_names.get(cls_id)
            if name is None:
                return None, f"unknown bean class {cls_id} in zone {zone}"
            digit = self._bean_color_code(name)
            if digit is None:
                return None, f"unmapped bean class '{name}' in zone {zone}"
            target_digits[zone] = digit

        digit_best, err = self._aggregate_digits(digit_dets)
        if err:
            return None, err

        placement_for_digit = {}
        for zone, det in digit_best.items():
            if det is None:
                continue
            try:
                cls_id = int(det[5])
            except (ValueError, TypeError):
                return None, f"invalid digit class in zone {zone}"
            name = self.digit_names.get(cls_id)
            if name is None:
                return None, f"unknown digit class {cls_id} in zone {zone}"
            try:
                value = int(name)
            except ValueError:
                return None, f"non-numeric digit name '{name}' in zone {zone}"
            if value not in (1, 2, 3):
                continue
            if value in placement_for_digit:
                return None, (
                    f"duplicate digit {value} in placement zones "
                    f"{placement_for_digit[value]} and {zone}"
                )
            placement_for_digit[value] = zone

        result = []
        for zone in (1, 2, 3):
            target = target_digits[zone]
            place = placement_for_digit.get(target)
            if place is None:
                return None, f"missing digit {target} in placement zones"
            result.append(place)

        return tuple(result), "ok"

    def generate_frame(self, bean_dets, digit_dets):
        """Build the new protocol result frame payload.

        Returns (c1, c2, c3, n4, n5, n6, n7, n8), "ok" on success, or
        (None, reason) on failure.

        c1-c3 are pickup-zone colors (1=yellow, 2=green, 3=white).
        n4-n8 are the digits placed in placement zones 4-8.
        """
        err = self._validate_lines()
        if err:
            return None, err

        for name, value in (
            ("mapping", self.mapping),
            ("bean_names", self.bean_names),
            ("digit_names", self.digit_names),
        ):
            if not isinstance(value, dict):
                return None, f"invalid {name}"

        bean_best, err = self._aggregate_beans(bean_dets)
        if err:
            return None, err

        colors = {}
        for zone in (1, 2, 3):
            det = bean_best[zone]
            if det is None:
                return None, f"missing bean in pickup zone {zone}"
            try:
                cls_id = int(det[5])
            except (ValueError, TypeError):
                return None, f"invalid bean class in zone {zone}"
            name = self.bean_names.get(cls_id)
            if name is None:
                return None, f"unknown bean class {cls_id} in zone {zone}"
            code = self._bean_color_code(name)
            if code is None:
                return None, f"unmapped bean class '{name}' in zone {zone}"
            colors[zone] = code

        if sorted(colors.values()) != [1, 2, 3]:
            return None, f"pickup colors must be 1,2,3 exactly once, got {list(colors.values())}"

        digit_best, err = self._aggregate_digits(digit_dets)
        if err:
            return None, err

        digits = {}
        for zone in (4, 5, 6, 7, 8):
            det = digit_best[zone]
            if det is None:
                return None, f"missing digit in placement zone {zone}"
            try:
                cls_id = int(det[5])
            except (ValueError, TypeError):
                return None, f"invalid digit class in zone {zone}"
            name = self.digit_names.get(cls_id)
            if name is None:
                return None, f"unknown digit class {cls_id} in zone {zone}"
            try:
                value = int(name)
            except ValueError:
                return None, f"non-numeric digit name '{name}' in zone {zone}"
            if not (1 <= value <= 5):
                return None, f"digit value {value} in zone {zone} out of range 1..5"
            digits[zone] = value

        if sorted(digits.values()) != [1, 2, 3, 4, 5]:
            return None, (
                f"placement digits must be 1..5 exactly once, got {list(digits.values())}"
            )

        payload = (
            colors[1],
            colors[2],
            colors[3],
            digits[4],
            digits[5],
            digits[6],
            digits[7],
            digits[8],
        )
        return payload, "ok"

    def generate_frame_with_completion(self, bean_dets, digit_dets, *,
                                       confidence_threshold=0.5,
                                       max_inferred_beans=1,
                                       max_inferred_digits=2):
        """Build result frame with logical completion of missing/duplicate zones.

        Unlike :meth:`generate_frame`, this method tolerates missing zones
        and duplicate classifications by inferring the most likely complete
        assignment via confidence-weighted permutation scoring.

        Returns:
            ``(payload, bean_results, digit_results, reason)`` where
            *payload* is an 8-tuple or ``None``, and *results* dicts map
            ``zone -> (value, 'detected'|'inferred', confidence)``.
        """
        from models.logic_completion import count_inferred, validate_and_complete

        err = self._validate_lines()
        if err:
            return None, {}, {}, err

        for name, value in (
            ("mapping", self.mapping),
            ("bean_names", self.bean_names),
            ("digit_names", self.digit_names),
        ):
            if not isinstance(value, dict):
                return None, {}, {}, f"invalid {name}"

        # ── aggregate detections ───────────────────────────────────────
        bean_best, err = self._aggregate_beans(bean_dets)
        if err:
            return None, {}, {}, err
        digit_best, err = self._aggregate_digits(digit_dets)
        if err:
            return None, {}, {}, err

        # ── map bean zones to (color_code, confidence) ─────────────────
        bean_zone_vals = {}
        for zone in (1, 2, 3):
            det = bean_best[zone]
            if det is None:
                bean_zone_vals[zone] = None
                continue
            cls_id = int(det[5])
            name = self.bean_names.get(cls_id)
            if name is None:
                bean_zone_vals[zone] = None
                continue
            code = self._bean_color_code(name)
            if code is None:
                bean_zone_vals[zone] = None
                continue
            bean_zone_vals[zone] = (code, float(det[4]))

        # ── map digit zones to (digit_value, confidence) ───────────────
        digit_zone_vals = {}
        for zone in (4, 5, 6, 7, 8):
            det = digit_best[zone]
            if det is None:
                digit_zone_vals[zone] = None
                continue
            cls_id = int(det[5])
            name = self.digit_names.get(cls_id)
            if name is None:
                digit_zone_vals[zone] = None
                continue
            try:
                val = int(name)
            except ValueError:
                digit_zone_vals[zone] = None
                continue
            if not (1 <= val <= 5):
                digit_zone_vals[zone] = None
                continue
            digit_zone_vals[zone] = (val, float(det[4]))

        # ── logical completion ─────────────────────────────────────────
        bean_results = validate_and_complete(
            bean_zone_vals, {1, 2, 3},
            confidence_threshold=confidence_threshold,
        )
        digit_results = validate_and_complete(
            digit_zone_vals, {1, 2, 3, 4, 5},
            confidence_threshold=confidence_threshold,
        )

        # ── guard: too many inferred zones → reject ────────────────────
        bean_inferred = count_inferred(bean_results)
        digit_inferred = count_inferred(digit_results)
        if bean_inferred > max_inferred_beans:
            return (None, bean_results, digit_results,
                    f"too many inferred bean zones "
                    f"({bean_inferred} > {max_inferred_beans})")
        if digit_inferred > max_inferred_digits:
            return (None, bean_results, digit_results,
                    f"too many inferred digit zones "
                    f"({digit_inferred} > {max_inferred_digits})")

        # ── build 8-tuple payload ──────────────────────────────────────
        payload = (
            bean_results[1][0], bean_results[2][0], bean_results[3][0],
            digit_results[4][0], digit_results[5][0], digit_results[6][0],
            digit_results[7][0], digit_results[8][0],
        )
        return payload, bean_results, digit_results, "ok"
