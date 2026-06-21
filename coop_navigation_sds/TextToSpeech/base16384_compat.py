"""Minimal pure-Python Base16384 codec used when the native wheel is unavailable."""

_DATA_OFFSET = 0x4E00
_REMAINDER_OFFSET = 0x3D00


def encode_to_string(data):
    """Encode bytes using the Base16384 representation consumed by ChatTTS."""
    payload = bytes(data)
    remainder = len(payload) % 7
    encoded = []
    for start in range(0, len(payload), 7):
        block = payload[start:start + 7]
        value = int.from_bytes(block, "big") << ((7 - len(block)) * 8)
        symbol_count = (len(block) * 8 + 13) // 14
        for index in range(symbol_count):
            shift = 56 - (index + 1) * 14
            encoded.append(chr(_DATA_OFFSET + ((value >> shift) & 0x3FFF)))
    if remainder:
        encoded.append(chr(_REMAINDER_OFFSET + remainder))
    return "".join(encoded)


def decode_from_string(data):
    """Decode a Base16384 string into bytes."""
    symbols = list(str(data))
    remainder = 0
    if symbols and _REMAINDER_OFFSET < ord(symbols[-1]) < _REMAINDER_OFFSET + 7:
        remainder = ord(symbols.pop()) - _REMAINDER_OFFSET

    decoded = bytearray()
    for start in range(0, len(symbols), 4):
        block = symbols[start:start + 4]
        value = 0
        for symbol in block:
            code = ord(symbol) - _DATA_OFFSET
            if not 0 <= code <= 0x3FFF:
                raise ValueError("Invalid Base16384 symbol.")
            value = (value << 14) | code
        value <<= (4 - len(block)) * 14
        is_final_block = start + len(block) == len(symbols)
        byte_count = remainder if is_final_block and remainder else 7
        decoded.extend(value.to_bytes(7, "big")[:byte_count])
    return bytes(decoded)
