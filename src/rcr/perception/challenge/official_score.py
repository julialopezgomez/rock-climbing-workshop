"""Reproduce the comparison score in rcr-perception at b8c607e.

The upstream matcher is an ordered augmenting-path assignment, not a minimum
cost assignment. Keep input order. Angles are wrapped Euler-component errors;
they are not SO(3) geodesic errors. Evaluation must run after inference.
"""

import math

SOURCE_COMMIT = "b8c607e8dc2749ae2f512ce7265ef3018f32e9f5"


def position_matches(active, test):
    def distance(a, b):
        return math.dist([a["position"][k] for k in "xyz"], [b["position"][k] for k in "xyz"])

    candidates = []
    for item in test:
        choices = [(distance(item, other), i) for i, other in enumerate(active)]
        candidates.append([i for d, i in sorted(choices) if d <= 0.1 + 1e-9])
    assigned = [-1] * len(active)

    def assign(j, visited):
        for i in candidates[j]:
            if i in visited:
                continue
            visited.add(i)
            if assigned[i] == -1 or assign(assigned[i], visited):
                assigned[i] = j
                return True
        return False

    for j in range(len(test)):
        assign(j, set())
    return [(i, j) for i, j in enumerate(assigned) if j != -1]


def _items(document, volumes):
    holds = document.get("holds", document)
    items = [{**hold, "name": name} for name, hold in holds.items()]
    result = {"holds": items}
    if volumes:
        if document.get("formatVersion") != 1:
            raise ValueError(
                "Volume scoring requires a full scene; never inherit reference volumes"
            )
        result["volumes"] = [
            {**v, "position": {**v["center"], "z": 0.0}} for v in document["scene"]["volumes"]
        ]
    for category in result.values():
        for item in category:
            values = list(item["position"].values()) + list(item.get("orientation", {}).values())
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError("Scoring requires finite numeric coordinates")
    return result


def score_scene(active, test, *, volumes=False):
    """Active is the reference scene; test is the serialized prediction.

    The final badge doubles matched errors (one copy on each side), and adds
    10 for every unmatched object. Colors, dimensions and runtime are unscored.
    Volume scoring is off by default in the current browser UI.
    """
    a, b = _items(active, volumes), _items(test, volumes)
    result = dict(source_commit=SOURCE_COMMIT, volume_scoring=volumes, final_score=0.0)
    for category in a:
        pairs = position_matches(a[category], b[category])
        rows = []
        for i, j in pairs:
            first, second = a[category][i], b[category][j]
            distance = math.dist(
                [first["position"][k] for k in "xyz"], [second["position"][k] for k in "xyz"]
            )
            position = 2 * (distance / 0.1) ** 2
            orientation, mismatch = 0.0, 0.0
            if category == "holds":
                differences = [
                    first["orientation"][k] - second["orientation"][k]
                    for k in ("roll", "pitch", "yaw")
                ]
                orientation = sum(
                    (math.atan2(math.sin(d), math.cos(d)) / (math.pi / 2)) ** 2 for d in differences
                )
                mismatch = 0.0 if first["type"] == second["type"] else 2.0
            rows.append(
                dict(
                    active=first["name"],
                    test=second["name"],
                    distance_m=distance,
                    position_score=2 * position,
                    orientation_score=2 * orientation,
                    type_score=2 * mismatch,
                    score=2 * (position + orientation + mismatch),
                )
            )
        matched_a, matched_b = {i for i, _ in pairs}, {j for _, j in pairs}
        missing = [h["name"] for i, h in enumerate(a[category]) if i not in matched_a]
        extra = [h["name"] for i, h in enumerate(b[category]) if i not in matched_b]
        total = sum(row["score"] for row in rows) + 10 * (len(missing) + len(extra))
        result[category] = dict(
            active_count=len(a[category]),
            test_count=len(b[category]),
            matches=len(pairs),
            missing=missing,
            extra=extra,
            pairs=rows,
            final_score=total,
        )
        result["final_score"] += total
    return result
