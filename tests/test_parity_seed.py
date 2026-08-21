import numpy as np

import eval_highway


def test_seed_proof_accepts_only_close_state_branch_straddles():
    below = np.nextafter(np.float32(3.0), np.float32(0.0))
    above = np.nextafter(np.float32(3.0), np.float32(np.inf))
    common = {
        "weights": np.array([1.0], dtype=np.float32),
        "vars": np.array([1.0], dtype=np.float32),
        "nmodes": 1,
    }
    prev_a = {**common, "means": np.array([[below]], dtype=np.float32)}
    prev_b = {**common, "means": np.array([[above]], dtype=np.float32)}

    proven, lines = eval_highway._seed_proof(
        prev_a, prev_b, np.array([0.0], dtype=np.float32),
        np.float32(16.0), np.float32(9.0), np.float32(0.9),
    )
    assert proven
    assert any("Tg" in line for line in lines)

    far_b = {**prev_b, "means": prev_b["means"] + np.float32(10.0)}
    proven, _ = eval_highway._seed_proof(
        prev_a, far_b, np.array([0.0], dtype=np.float32),
        np.float32(16.0), np.float32(9.0), np.float32(0.9),
    )
    assert not proven

    unreachable_a = {
        "weights": np.array([0.8, 0.2], dtype=np.float32),
        "means": np.array([[0.0], [below]], dtype=np.float32),
        "vars": np.ones(2, dtype=np.float32),
        "nmodes": 2,
    }
    unreachable_b = {
        **unreachable_a,
        "means": np.array([[0.0], [above]], dtype=np.float32),
    }
    proven, _ = eval_highway._seed_proof(
        unreachable_a, unreachable_b, np.array([0.0], dtype=np.float32),
        np.float32(16.0), np.float32(9.0), np.float32(0.9),
    )
    assert not proven
