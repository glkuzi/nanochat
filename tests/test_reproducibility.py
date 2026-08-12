"""Tests for centralized reproducibility controls."""

import os
import random

import numpy as np
import torch

from nanochat.common import configure_reproducibility
import nanochat.flash_attention as flash_attention


def test_configure_reproducibility_reseeds_all_rngs():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        configure_reproducibility(seed=12345, deterministic=True)
        first = (random.random(), np.random.random(), torch.rand(4))

        configure_reproducibility(seed=12345, deterministic=True)
        second = (random.random(), np.random.random(), torch.rand(4))

        assert first[0] == second[0]
        assert first[1] == second[1]
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["PYTHONHASHSEED"] == "12345"
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic)


def test_configure_reproducibility_rejects_invalid_seed():
    try:
        configure_reproducibility(seed=-1)
    except ValueError as exc:
        assert str(exc) == "seed must be a non-negative integer"
    else:
        raise AssertionError("negative seed should fail")


def test_strict_determinism_disables_fa3_runtime_path():
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_use_fa3 = flash_attention.USE_FA3
    try:
        flash_attention.USE_FA3 = True
        torch.use_deterministic_algorithms(True)
        assert not flash_attention.is_fa3_active()
        torch.use_deterministic_algorithms(False)
        assert flash_attention.is_fa3_active()
    finally:
        flash_attention.USE_FA3 = previous_use_fa3
        torch.use_deterministic_algorithms(previous_deterministic)
