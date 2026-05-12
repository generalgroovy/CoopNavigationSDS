"""Station-name model. It provides deterministic readable station names for generated transit networks.
"""
STATION_NAMES = [
    "Alpha",
    "Bravo",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
    "India",
    "Juliett",
    "Kilo",
    "Lima",
    "Mike",
    "November",
    "Oscar",
    "Papa",
    "Quebec",
    "Romeo",
    "Sierra",
    "Tango",
    "Uniform",
    "Victor",
    "Whiskey",
    "Xray",
    "Yankee",
    "Zulu",
]


def get_station_names(count: int):
    """Get station names function for this module's MVC responsibility.
    
    Args:
        count: Input value used by `get_station_names`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    if count > len(STATION_NAMES):
        raise ValueError(f"Only {len(STATION_NAMES)} station names available.")
    return STATION_NAMES[:count]
