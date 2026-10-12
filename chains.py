import numpy as np


def ssh(N, t1, t2, periodic=False):
    """SSH chain with N cells, sites ordered A1, B1, A2, B2, ..."""
    H = np.zeros((2 * N, 2 * N))
    for n in range(N):
        H[2 * n, 2 * n + 1] = t1
        H[2 * n + 1, 2 * n] = t1
    for n in range(N - 1):
        H[2 * n + 1, 2 * n + 2] = t2
        H[2 * n + 2, 2 * n + 1] = t2
    if periodic:
        H[2 * N - 1, 0] = t2
        H[0, 2 * N - 1] = t2
    return H


def winding_number(z):
    """Number of times the closed curve z(k), k from 0 to 2*pi, winds around the origin."""
    phase = np.unwrap(np.angle(z))
    return (phase[-1] - phase[0]) / (2 * np.pi)


def hatano_nelson(N, t, g, periodic=False):
    """Hatano-Nelson chain: hopping t + g from site n to n+1, t - g from n+1 to n."""
    H = np.zeros((N, N))
    for n in range(N - 1):
        H[n + 1, n] = t + g
        H[n, n + 1] = t - g
    if periodic:
        H[0, N - 1] = t + g
        H[N - 1, 0] = t - g
    return H


def nh_ssh(N, t1, t2, gamma, periodic=False):
    """Non-Hermitian SSH (Yao and Wang): t1 + gamma/2 from B_n to A_n, t1 - gamma/2 from A_n to B_n."""
    H = np.zeros((2 * N, 2 * N))
    for n in range(N):
        H[2 * n, 2 * n + 1] = t1 + gamma / 2
        H[2 * n + 1, 2 * n] = t1 - gamma / 2
    for n in range(N - 1):
        H[2 * n + 1, 2 * n + 2] = t2
        H[2 * n + 2, 2 * n + 1] = t2
    if periodic:
        H[2 * N - 1, 0] = t2
        H[0, 2 * N - 1] = t2
    return H
