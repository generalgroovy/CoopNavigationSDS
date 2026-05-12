"""Route proposal memory and duplicate-detection helpers for one dialog."""


class RouteProposalMemory:
    """Tracks spoken route candidates so a dialog does not accept repeats."""

    def __init__(self):
        self.seen = set()
        self.candidates = []

    def key(self, route):
        return tuple(route)

    def already_seen(self, route):
        return self.key(route) in self.seen

    def record(self, turn, route, duration, best_duration):
        previous_best = best_duration
        if best_duration is None:
            decision = "baseline"
        elif duration < best_duration:
            decision = "improved"
        elif duration == best_duration:
            decision = "tied"
        else:
            decision = "slower"

        candidate = {
            "turn": turn,
            "route": route,
            "duration": duration,
            "decision": decision,
            "best_duration": min(
                value for value in (best_duration, duration) if value is not None
            ),
            "previous_best": previous_best,
        }
        self.seen.add(self.key(route))
        self.candidates.append(candidate)
        return candidate
