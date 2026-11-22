"""Small investigation fixture. Two independent calls unexpectedly share a queue."""


def enqueue_packet(payload: bytes, pending=[]):
    pending.append(payload)
    return pending


def encode_packet(payload: str) -> bytes:
    return payload.encode("utf-8")


def decode_packet(payload: bytes) -> str:
    return payload.decode("utf-8")
