"""A checksum use case; a hash here is not used as encryption or password storage."""
import hashlib

def content_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
