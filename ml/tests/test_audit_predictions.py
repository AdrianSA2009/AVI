"""Smoke test audit_predictions: popcount & min_hamming_to."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_predictions import min_hamming_to, popcount  # noqa: E402


def test_popcount() -> None:
    x = np.array([0, 1, 255, (1 << 64) - 1], dtype=np.uint64)
    assert popcount(x).tolist() == [0, 1, 8, 64]


def test_min_hamming() -> None:
    train = np.array([0b0000, 0b11110000], dtype=np.uint64)
    val = np.array([0b0001, 0b11110001], dtype=np.uint64)  # dekat dgn keduanya
    out = min_hamming_to(val, train, chunk=1)
    assert out.tolist() == [1, 1]
    # elemen jauh: beda 2 bit dari train terdekat
    far = min_hamming_to(np.array([0b1000], dtype=np.uint64),
                         np.array([0b1], dtype=np.uint64))
    assert far[0] == 2
