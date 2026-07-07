from models.mapping import bean_name_to_digit


class TaskGenerator:
    """Convert bean/digit detections into three placement-zone numbers."""

    _BEAN_KEYS = ("left_x", "right_x")
    _DIGIT_KEYS = ("horizontal_y", "left_x", "right_x")

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
        d = self.lines["digit"]
        horizontal_y = d["horizontal_y"]
        left_x = d["left_x"]
        right_x = d["right_x"]
        midpoint = (left_x + right_x) / 2

        if cy < horizontal_y:
            # Strict < on midpoint: detections exactly on midpoint go to zone 8.
            return 4 if cx < midpoint else 8
        if cx < left_x:
            return 5
        if cx < right_x:
            return 6
        return 7

    def _center(self, det):
        x1, y1, x2, y2 = det[:4]
        return (x1 + x2) / 2, (y1 + y2) / 2

    def generate(self, bean_dets, digit_dets):
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

        # Aggregate best bean per pickup zone
        bean_best = {1: None, 2: None, 3: None}
        for det in bean_dets:
            err = self._validate_detection(det)
            if err:
                return None, err
            cx, cy = self._center(det)
            zone = self._bean_zone(cx)
            if bean_best[zone] is None or det[4] > bean_best[zone][4]:
                bean_best[zone] = det

        # Map beans to target digits
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
            digit = bean_name_to_digit(name, self.mapping)
            if digit is None:
                return None, f"unmapped bean class '{name}' in zone {zone}"
            target_digits[zone] = digit

        # Aggregate best digit per placement zone
        digit_best = {4: None, 5: None, 6: None, 7: None, 8: None}
        for det in digit_dets:
            err = self._validate_detection(det)
            if err:
                return None, err
            cx, cy = self._center(det)
            zone = self._digit_zone(cx, cy)
            if digit_best[zone] is None or det[4] > digit_best[zone][4]:
                digit_best[zone] = det

        # Reverse lookup: target digit -> placement zone; detect duplicates
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
            # Only digits 1-3 participate in task generation; 4/5 are empty boxes.
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
