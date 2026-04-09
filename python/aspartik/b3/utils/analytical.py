import numpy as np


def _build_nuc_table() -> np.ndarray:
    result = np.zeros(256, dtype=np.intp)
    result[ord("A")] = 0
    result[ord("C")] = 1
    result[ord("G")] = 2
    result[ord("T")] = 3
    return result


_NUC_TABLE = _build_nuc_table()


def _encode(seq: str) -> np.ndarray:
    return _NUC_TABLE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def _jc_matrix(t: float) -> np.ndarray:
    exp = np.exp(-4 * t / 3)
    same = 0.25 + 0.75 * exp
    diff = 0.25 - 0.25 * exp
    P = np.full((4, 4), diff)
    np.fill_diagonal(P, same)
    return P


def _hky_matrix(kappa: float, freqs: tuple[float, ...], t: float) -> np.ndarray:
    pi_a, pi_c, pi_g, pi_t = freqs
    r = pi_a + pi_g
    y = pi_c + pi_t

    div = 2.0 * (
        pi_g * pi_t
        + pi_a * pi_c
        + pi_a * pi_t
        + pi_c * pi_g
        + kappa * (pi_a * pi_g + pi_c * pi_t)
    )

    D = np.array([0.0, -1.0 / div, -(y + r * kappa) / div, -(r + y * kappa) / div])

    P = np.array(
        [
            [1.0, -y / r, -pi_g / pi_a, 0.0],
            [1.0, 1.0, 0.0, -pi_t / pi_c],
            [1.0, -y / r, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
        ]
    )

    inv_P = np.array(
        [
            [pi_a, pi_c, pi_g, pi_t],
            [-pi_a, pi_c * r / y, -pi_g, pi_t * r / y],
            [-pi_a / r, 0.0, pi_a / r, 0.0],
            [0.0, -pi_c / y, 0.0, pi_c / y],
        ]
    )

    # P @ diag(exp(D*t)) @ inv_P
    return (P * np.exp(D * t)) @ inv_P


def analytical_jc_2leaf(seq1: str, seq2: str, branch_len: float) -> float:
    s1, s2 = _encode(seq1), _encode(seq2)
    P = _jc_matrix(branch_len)
    p1 = P[:, s1]
    p2 = P[:, s2]
    site_l = (0.25 * p1 * p2).sum(axis=0)
    return np.log(site_l).sum()


def analytical_hky_2leaf(
    seq1: str,
    seq2: str,
    branch_len: float,
    kappa: float,
    freqs: tuple[float, ...],
) -> float:
    s1, s2 = _encode(seq1), _encode(seq2)
    tm = _hky_matrix(kappa, freqs, branch_len)
    pi = np.array(freqs)
    site_l = (pi[:, np.newaxis] * tm[:, s1] * tm[:, s2]).sum(axis=0)
    return np.log(site_l).sum()
