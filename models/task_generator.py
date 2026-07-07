class TaskGenerator:
    """Convert bean/digit detections into three placement-zone numbers."""

    def __init__(self, mapping, lines, digit_names):
        self.mapping = mapping
        self.lines = lines
        self.digit_names = digit_names

    def _bean_zone(self, cx):
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
        # Aggregate best bean per pickup zone
        bean_best = {1: None, 2: None, 3: None}
        for det in bean_dets:
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
            cls_id = int(det[5])
            # bean config names: 0:white, 1:green, 2:yellow
            name_map = {0: "white", 1: "green", 2: "yellow"}
            name = name_map.get(cls_id)
            digit = self.mapping.get(name)
            if digit is None:
                return None, f"unknown bean class {cls_id} in zone {zone}"
            target_digits[zone] = digit

        # Aggregate best digit per placement zone
        digit_best = {4: None, 5: None, 6: None, 7: None, 8: None}
        for det in digit_dets:
            cx, cy = self._center(det)
            zone = self._digit_zone(cx, cy)
            if digit_best[zone] is None or det[4] > digit_best[zone][4]:
                digit_best[zone] = det

        # Reverse lookup: target digit -> placement zone
        placement_for_digit = {}
        for zone, det in digit_best.items():
            if det is None:
                continue
            cls_id = int(det[5])
            name = self.digit_names.get(cls_id)
            if name is None:
                continue
            try:
                value = int(name)
            except ValueError:
                continue
            placement_for_digit[value] = zone

        result = []
        for zone in (1, 2, 3):
            target = target_digits[zone]
            place = placement_for_digit.get(target)
            if place is None:
                return None, f"missing digit {target} in placement zones"
            result.append(place)

        return tuple(result), "ok"
