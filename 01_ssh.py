import numpy as np
import matplotlib.pyplot as plt
from chains import ssh, winding_number

t2 = 1.0
cases = [0.5, 1.0, 1.5]   # t1 values: t1 < t2, t1 = t2, t1 > t2
N = 40                    # unit cells for the open chain
k = np.linspace(0, 2 * np.pi, 801)


def h(k, t1, t2):
    return t1 + t2 * np.exp(-1j * k)


# Bloch bands
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, t1 in zip(axes, cases):
    E = np.abs(h(k, t1, t2))
    ax.plot(k, E, 'C0', label='upper band')
    ax.plot(k, -E, 'C1', label='lower band')
    ax.set_title(f'$t_1/t_2 = {t1 / t2:g}$')
    ax.set_xlabel('$k$')
    ax.set_xticks([0, np.pi, 2 * np.pi], ['0', r'$\pi$', r'$2\pi$'])
axes[0].set_ylabel('$E / t_2$')
axes[0].legend()
fig.tight_layout()
fig.savefig('figures/ssh_bands.png', dpi=200)

# check the Bloch formula against the periodic chain
for t1 in cases:
    E_chain = np.sort(np.linalg.eigvalsh(ssh(N, t1, t2, periodic=True)))
    kn = 2 * np.pi * np.arange(N) / N
    E_bloch = np.sort(np.concatenate([np.abs(h(kn, t1, t2)), -np.abs(h(kn, t1, t2))]))
    assert np.allclose(E_chain, E_bloch)
print('periodic chain matches +-|t1 + t2 e^(-ik)|')

# h(k) in the complex plane and its winding number
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, t1 in zip(axes, cases):
    z = h(k, t1, t2)
    ax.plot(z.real, z.imag, label='$h(k)$')
    ax.plot(0, 0, 'kx', ms=10, mew=2, label='origin')
    if t1 == t2:
        nu = 'undefined (curve touches 0)'
    else:
        nu = f'{winding_number(z):.0f}'
    print(f't1/t2 = {t1 / t2:g}: winding number = {winding_number(z):+.4f}')
    ax.set_title(f'$t_1/t_2 = {t1 / t2:g}$, winding = {nu}', fontsize=10)
    ax.set_xlabel('Re $h$')
    ax.set_ylabel('Im $h$')
    ax.set_aspect('equal')
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
axes[0].legend(loc='upper right')
fig.tight_layout()
fig.savefig('figures/ssh_winding.png', dpi=200)

# open chain spectrum against t1/t2
t1_list = np.linspace(0, 2, 201)
E_open = np.array([np.linalg.eigvalsh(ssh(N, t1, t2)) for t1 in t1_list])
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(t1_list, E_open, 'k', lw=0.5)
# the two states closest to zero
ax.plot(t1_list, E_open[:, N - 1], 'r', lw=1.5, label='two states closest to $E=0$')
ax.plot(t1_list, E_open[:, N], 'r', lw=1.5)
ax.axvline(1, color='grey', ls='--', lw=0.8)
ax.set_xlabel('$t_1 / t_2$')
ax.set_ylabel('$E / t_2$')
ax.set_title(f'Open SSH chain, N = {N} cells')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig('figures/ssh_open_spectrum.png', dpi=200)

# zero modes for t1 < t2
t1 = 0.5
E, psi = np.linalg.eigh(ssh(N, t1, t2))
print(f't1/t2 = {t1}: two energies closest to zero = {E[N - 1]:.2e}, {E[N]:.2e}')
assert abs(E[N - 1]) < 1e-8 and abs(E[N]) < 1e-8

# the left and right edge states overlap a little, which splits them to
# +-(t2 - t1^2/t2) (t1/t2)^N (derived in NOTES.md)
E_split = (t2 - t1 ** 2 / t2) * (t1 / t2) ** N
print(f'predicted splitting +-{E_split:.3e}, numerical +-{E[N]:.3e}')
assert np.isclose(E[N], E_split, rtol=1e-3)

# The two near-zero eigenstates are the even and odd mixtures of a left and a
# right edge state (they split by ~ (t1/t2)^N because the edges overlap slightly).
# Chiral symmetry means the left state lives only on A sites and the right state
# only on B sites, so the A and B parts of either eigenstate separate the edges.
psi0 = psi[:, N]
psi_left = psi0[0::2] / np.linalg.norm(psi0[0::2])
psi_right = psi0[1::2] / np.linalg.norm(psi0[1::2])
cells = np.arange(1, N + 1)

ratio = np.abs(psi_left[1:6] / psi_left[:5])
print(f'|psi_A(n+1) / psi_A(n)| for first cells: {ratio}, expected t1/t2 = {t1 / t2}')
assert np.allclose(ratio, t1 / t2, rtol=1e-6)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.semilogy(cells, np.abs(psi_left) ** 2, 'o', ms=4, label='left edge state (A sites)')
ax.semilogy(cells, np.abs(psi_right) ** 2, 's', ms=4, label='right edge state (B sites)')
ax.semilogy(cells, np.abs(psi_left[0]) ** 2 * (t1 / t2) ** (2 * (cells - 1)), 'k--', lw=1,
            label=r'$(t_1/t_2)^{2(n-1)}$')
ax.semilogy(cells, np.abs(psi_right[-1]) ** 2 * (t1 / t2) ** (2 * (N - cells)), 'k:', lw=1,
            label=r'$(t_1/t_2)^{2(N-n)}$')
ax.set_xlabel('unit cell $n$')
ax.set_ylabel(r'$|\psi|^2$')
ax.set_title(f'Zero modes, $t_1/t_2 = {t1}$, N = {N}')
ax.legend()
fig.tight_layout()
fig.savefig('figures/ssh_edge_states.png', dpi=200)
