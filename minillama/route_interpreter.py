"""Route interpretation model that infers connected station sequences from natural Agent B replies.
"""
import re

from minillama.agents import STATION_LOOKUP, STATION_PATTERN
from minillama.config import ROUTE_INTERPRETER_SCORING
from minillama.route_planner import estimate_route_time, route_is_valid


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
        """Station mentions method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `station_mentions`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        result = []
        for match in STATION_PATTERN.finditer(text):
            station = STATION_LOOKUP[match.group(1).lower()]
            if not result or result[-1] != station:
                result.append(station)
        return result

    def interpret_reply(self, text, scenario):
        """Interpret reply method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `interpret_reply`; see the function signature and caller context for the expected type.
            scenario: Input value used by `interpret_reply`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
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
        """Has station mentions method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `has_station_mentions`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return bool(STATION_PATTERN.search(text))

    def _candidate_fragments(self, text):
        """ candidate fragments method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `_candidate_fragments`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
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
        """ valid subroutes method for this module's MVC responsibility.
        
        Args:
            mentions: Input value used by `_valid_subroutes`; see the function signature and caller context for the expected type.
            fragment: Input value used by `_valid_subroutes`; see the function signature and caller context for the expected type.
            fragment_index: Input value used by `_valid_subroutes`; see the function signature and caller context for the expected type.
            scenario: Input value used by `_valid_subroutes`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        candidates = []
        lower = fragment.lower()
        marker_bonus = (
            ROUTE_INTERPRETER_SCORING["marker_bonus"]
            if any(marker in lower for marker in NATURAL_ROUTE_MARKERS)
            else 0
        )

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
                    fragment_index * ROUTE_INTERPRETER_SCORING["fragment_index_weight"]
                    + marker_bonus
                    + len(route) * ROUTE_INTERPRETER_SCORING["route_length_weight"]
                    + (ROUTE_INTERPRETER_SCORING["starts_correctly_bonus"] if starts_correctly else 0)
                    + (ROUTE_INTERPRETER_SCORING["reaches_goal_bonus"] if reaches_goal else 0)
                    - arrival / ROUTE_INTERPRETER_SCORING["arrival_penalty_divisor"]
                )

                candidates.append({"route": route, "score": score})

        return candidates
