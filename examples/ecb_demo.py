"""Intentionally flawed, authored fixture for isolated crypto-tool demonstrations."""
from Crypto.Cipher import AES

def encrypt_demo(data: bytes) -> bytes:
    demo_key = b"demonstrationkey"  # 16-byte fixture, not a real credential.
    return AES.new(demo_key, AES.MODE_ECB).encrypt(data)
