"""Route interpretation model that infers connected station sequences from natural Agent B replies.
"""
import re

from minillama.agent_a.agents import STATION_LOOKUP, STATION_PATTERN
from minillama.evaluation.config import ROUTE_INTERPRETER_SCORING
from minillama.model.metro_data import ADJACENCY, LINES
from minillama.model.route_planner import estimate_route_time, line_direction_sequences, route_is_valid, segment_travel


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
        station_route = self._route_from_labeled_station_sequence(text, "Stations")
        if station_route and route_is_valid(station_route):
            if station_route[0] == scenario["start_station"] and station_route[-1] == scenario["destination_station"]:
                return station_route
        line_boarding_route = self._route_from_boarding_and_lines(text)
        if line_boarding_route and route_is_valid(line_boarding_route):
            if line_boarding_route[0] == scenario["start_station"] and line_boarding_route[-1] == scenario["destination_station"]:
                return line_boarding_route
        boarding_route = self._route_from_labeled_station_sequence(text, "Boarding")
        if boarding_route and route_is_valid(boarding_route):
            if boarding_route[0] == scenario["start_station"] and boarding_route[-1] == scenario["destination_station"]:
                return boarding_route

        line_route = self._route_from_named_line_legs(text)
        if line_route and route_is_valid(line_route):
            if line_route[0] == scenario["start_station"] and line_route[-1] == scenario["destination_station"]:
                return line_route
            lower = text.lower()
            marker_bonus = (
                ROUTE_INTERPRETER_SCORING["marker_bonus"]
                if any(marker in lower for marker in NATURAL_ROUTE_MARKERS)
                else 0
            )
            estimate = estimate_route_time(
                line_route,
                scenario["start_time_min"],
                scenario["transfer_time_min"],
            )
            arrival = estimate[0] if estimate else 10**9
            candidates.append(
                {
                    "route": line_route,
                    "score": (
                        marker_bonus
                        + len(line_route) * ROUTE_INTERPRETER_SCORING["route_length_weight"]
                        + (ROUTE_INTERPRETER_SCORING["starts_correctly_bonus"] if line_route[0] == scenario["start_station"] else 0)
                        + (ROUTE_INTERPRETER_SCORING["reaches_goal_bonus"] if line_route[-1] == scenario["destination_station"] else 0)
                        - arrival / ROUTE_INTERPRETER_SCORING["arrival_penalty_divisor"]
                    ),
                }
            )

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

    def _route_from_labeled_station_sequence(self, text, label):
        """Parse an explicit station sequence such as 'Stations: Alpha to Bravo'."""
        pattern = re.compile(rf"\b{re.escape(label)}\s*:\s*([^.!?]+)", flags=re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return []
        mentions = self.station_mentions(match.group(1))
        if len(mentions) < 2:
            return []
        route = self._expand_spoken_route(mentions)
        return [] if len(route) != len(set(route)) else route

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
                route = self._expand_spoken_route(route)
                if len(route) != len(set(route)):
                    continue
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

    def _route_from_named_line_legs(self, text):
        """Parse compact spoken line legs such as 'Take Ring from A to B'."""
        legs = []
        station_group = r"\b(?:" + "|".join(re.escape(station) for station in STATION_LOOKUP.values()) + r")\b"
        for line_name in sorted(LINES, key=len, reverse=True):
            pattern = re.compile(
                rf"\b{re.escape(line_name)}\b\s+(?:from\s+)?({station_group})\s+to\s+({station_group})",
                flags=re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                origin = STATION_LOOKUP[match.group(1).lower()]
                target = STATION_LOOKUP[match.group(2).lower()]
                segment = self._line_segment_path(line_name, origin, target)
                if segment:
                    legs.append((match.start(), segment))

        if not legs:
            return []

        route = []
        for _, segment in sorted(legs, key=lambda item: item[0]):
            if not route:
                route.extend(segment)
            elif route[-1] == segment[0]:
                route.extend(segment[1:])
            else:
                bridge = self._shortest_station_path(route[-1], segment[0])
                if not bridge:
                    return []
                route.extend(bridge[1:])
                route.extend(segment[1:])
        return route

    def _route_from_boarding_and_lines(self, text):
        """Expand compact 'Boarding' routes using the spoken line names."""
        boarding_points = self._raw_labeled_station_mentions(text, "Boarding")
        if len(boarding_points) < 2:
            return []
        line_names = self._line_mentions(text)
        if not line_names:
            return []
        if len(line_names) == 1:
            line_names = line_names * (len(boarding_points) - 1)
        if len(line_names) < len(boarding_points) - 1:
            return []

        route = []
        for index, (origin, target) in enumerate(zip(boarding_points, boarding_points[1:])):
            segment = self._line_segment_path(line_names[index], origin, target)
            if not segment:
                return []
            if not route:
                route.extend(segment)
            elif route[-1] == segment[0]:
                route.extend(segment[1:])
            else:
                return []
        return [] if len(route) != len(set(route)) else route

    def _raw_labeled_station_mentions(self, text, label):
        pattern = re.compile(rf"\b{re.escape(label)}\s*:\s*([^.!?]+)", flags=re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return []
        return self.station_mentions(match.group(1))

    @staticmethod
    def _line_mentions(text):
        mentions = []
        for line_name in sorted(LINES, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(line_name)}\b", text, flags=re.IGNORECASE):
                mentions.append((match.start(), line_name))
        distinct = []
        for _, line_name in sorted(mentions, key=lambda item: item[0]):
            if not distinct or distinct[-1] != line_name:
                distinct.append(line_name)
        return distinct

    @staticmethod
    def _line_segment_path(line_name, origin, target):
        """Return the station path between two stops on the named line."""
        options = []
        for sequence in line_direction_sequences(line_name):
            if sequence and sequence[0] == sequence[-1]:
                base = sequence[:-1]
                if origin in base and target in base:
                    start_index = base.index(origin)
                    path = [origin]
                    for offset in range(1, len(base) + 1):
                        station = base[(start_index + offset) % len(base)]
                        path.append(station)
                        if station == target:
                            travel = sum(segment_travel(a, b) for a, b in zip(path, path[1:]))
                            options.append((travel, len(path), path))
                            break
                continue
            for start_index, station in enumerate(sequence):
                if station != origin:
                    continue
                for end_index in range(start_index + 1, len(sequence)):
                    if sequence[end_index] == target:
                        path = sequence[start_index:end_index + 1]
                        travel = sum(segment_travel(a, b) for a, b in zip(path, path[1:]))
                        options.append((travel, len(path), path))
        if not options:
            return []
        return min(options, key=lambda item: (item[0], item[1]))[2]

    def _expand_spoken_route(self, mentions):
        """Expand compact boarding/change mentions into adjacent station paths."""
        if route_is_valid(mentions):
            return mentions
        if len(mentions) < 2:
            return mentions

        expanded = [mentions[0]]
        for origin, target in zip(mentions, mentions[1:]):
            segment = self._shortest_station_path(origin, target)
            if not segment:
                return mentions
            expanded.extend(segment[1:])
        return expanded

    @staticmethod
    def _shortest_station_path(origin, target):
        """Find the shortest station-count path between two named stops."""
        if origin == target:
            return [origin]

        queue = [(origin, [origin])]
        seen = {origin}
        for station, path in queue:
            for nxt, _, _ in ADJACENCY[station]:
                if nxt in seen:
                    continue
                next_path = path + [nxt]
                if nxt == target:
                    return next_path
                seen.add(nxt)
                queue.append((nxt, next_path))
        return []
