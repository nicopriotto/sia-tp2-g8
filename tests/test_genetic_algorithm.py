def _calc_k_offspring(generational_gap: float, population_size: int) -> int:
    k = max(2, int(generational_gap * population_size))
    if k % 2 != 0:
        k += 1
    return k


def test_gap_g1_generates_n_children():
    k = _calc_k_offspring(1.0, 20)
    assert k == 20


def test_gap_g05_generates_half_children():
    k = _calc_k_offspring(0.5, 20)
    assert k == 10


def test_gap_minimum_two_children():
    k = _calc_k_offspring(0.01, 20)
    assert k >= 2
