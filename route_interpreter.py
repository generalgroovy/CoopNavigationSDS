import re

from agents import STATION_LOOKUP, STATION_PATTERN
from route_planner import estimate_route_time, route_is_valid


NATURAL_ROUTE_MARKERS = (
    "i recommend",
    "i would take",
    "you should take",
    "the quickest way",
    "the faster way",
    "the better option",
    "the route",
    "take",
    "go from",
    "continue",
    "change at",
)


class NaturalRouteInterpreter:
    """Infers proposed routes from ordinary spoken station mentions."""

    def station_mentions(self, text):
        result = []
        for match in STATION_PATTERN.finditer(text):
            station = STATION_LOOKUP[match.group(1).lower()]
            if not result or result[-1] != station:
                result.append(station)
        return result

    def interpret_reply(self, text, scenario):
        candidates = []

        fragments = self._candidate_fragments(text)
        for fragment_index, fragment in enumerate(fragments):
            mentions = self.station_mentions(fragment)
            candidates.extend(
                self._valid_subroutes(mentions, fragment, fragment_index, scenario)
            )

        if not candidates:
            mentions = self.station_mentions(text)
            candidates.extend(self._valid_subroutes(mentions, text, 0, scenario))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[0]["route"]

    def has_station_mentions(self, text):
        return bool(STATION_PATTERN.search(text))

    def _candidate_fragments(self, text):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]

        fragments = list(sentences)
        for start in range(len(sentences)):
            for end in range(start + 2, len(sentences) + 1):
                fragments.append(" ".join(sentences[start:end]))

        return fragments

    def _valid_subroutes(self, mentions, fragment, fragment_index, scenario):
        candidates = []
        lower = fragment.lower()
        marker_bonus = 30 if any(marker in lower for marker in NATURAL_ROUTE_MARKERS) else 0

        for start in range(len(mentions)):
            for end in range(start + 2, len(mentions) + 1):
                route = mentions[start:end]
                if not route_is_valid(route):
                    continue

                starts_correctly = route[0] == scenario["start_station"]
                reaches_goal = route[-1] == scenario["destination_station"]
                estimate = estimate_route_time(
                    route,
                    scenario["start_time_min"],
                    scenario["transfer_time_min"],
                )
                arrival = estimate[0] if estimate else 10**9

                score = (
                    fragment_index * 10
                    + marker_bonus
                    + len(route)
                    + (100 if starts_correctly else 0)
                    + (200 if reaches_goal else 0)
                    - arrival / 10000
                )

                candidates.append({"route": route, "score": score})

        return candidates
