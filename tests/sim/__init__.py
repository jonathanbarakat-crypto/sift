# Sim harness tests — always under SIFT_ENV=test (ephemeral test DB only).
import os

os.environ["SIFT_ENV"] = "test"
os.environ.setdefault("SIFT_CARD_HMAC_SECRET", "sim-test-hmac-secret")
