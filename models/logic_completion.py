"""Logical completion of missing/duplicate/ambiguous per-zone detections.

When the vision model misses a zone or produces a duplicate classification,
this module infers the most likely complete assignment using constraint
satisfaction and confidence scoring.
"""

import itertools


def validate_and_complete(zone_vals, required_set, *,
                          confidence_threshold=0.5):
    """Resolve missing/duplicate per-zone detections into a complete assignment.

    Args:
        zone_vals: {zone_id: (value, confidence)} — zones with no detection
            have entry ``None``.
        required_set: ``set`` of values that must appear exactly once across
            all zones (e.g. ``{1,2,3}`` for beans, ``{1,2,3,4,5}`` for digits).
        confidence_threshold: detections with confidence below this value are
            treated as uncertain but still participate in scoring.

    Returns:
        {zone_id: (value, 'detected'|'inferred', confidence)}
        Confidence is ``None`` for inferred zones.
    """
    zones = sorted(zone_vals.keys())
    required = sorted(required_set)

    if len(zones) != len(required):
        raise ValueError(
            f"zone count {len(zones)} must equal required count {len(required)}"
        )

    # Build confidence score matrix: score[zone][value] = confidence (or 0.0)
    score = {z: {v: 0.0 for v in required} for z in zones}
    empty_zones = set()
    for z in zones:
        entry = zone_vals.get(z)
        if entry is None:
            empty_zones.add(z)
            continue
        val, conf = entry
        if val in required_set:
            score[z][val] = float(conf)
        else:
            empty_zones.add(z)

    # Enumerate all permutations of required values across zones.
    # For 3 bean zones (6 perms) this is trivial.
    # For 5 digit zones (120 perms) this is still fast enough (< 1 ms).
    best_perm = None
    best_total = -1.0
    best_detected = set()

    for perm in itertools.permutations(required):
        total = sum(score[z][perm[i]] for i, z in enumerate(zones))
        if total > best_total:
            best_total = total
            best_perm = perm
            best_detected = {
                z for i, z in enumerate(zones)
                if score[z][perm[i]] > 0 and z not in empty_zones
            }

    # Build results
    results = {}
    for i, z in enumerate(zones):
        value = best_perm[i]
        if z in best_detected:
            entry = zone_vals[z]
            conf = float(entry[1]) if entry is not None else None
            results[z] = (value, "detected", conf)
        else:
            results[z] = (value, "inferred", None)

    return results


def count_inferred(results):
    """Return the number of zones with 'inferred' source."""
    return sum(1 for (_val, src, _conf) in results.values() if src == "inferred")
