import numpy as np
import matplotlib.pyplot as plt
from chains import nh_ssh, ssh, winding_number

t2 = 1.0
gamma = 4 / 3            # as in Yao and Wang
# N = 20 cells: eig on the open chain agrees with the exact spectrum to ~1e-13 here,
# but at N = 40 the error near t1 = gamma/2 grows to ~1e-5 (same non-normality as in Part 2).
N = 20
N_pbc = 100

t1_bloch = [t2 - gamma / 2, t2 + gamma / 2]      # Bloch gap closings
t1_obc = np.sqrt(t2 ** 2 + gamma ** 2 / 4)       # open-chain transition
print(f'Bloch gap closings at t1 = {t1_bloch[0]:.4f}, {t1_bloch[1]:.4f}; '
      f'open-chain transition at t1 = {t1_obc:.4f}')

# grid offset so it never lands exactly on t1 = gamma/2, where one hopping is zero
t1_list = np.linspace(0.001, 2.001, 401)


def add_lines(ax):
    ax.axvline(t1_bloch[0], color='C0', ls='--', lw=1, label=r'Bloch gap closings $t_2 \pm \gamma/2$')
    ax.axvline(t1_bloch[1], color='C0', ls='--', lw=1)
    ax.axvline(t1_obc, color='C3', ls='-', lw=1, label=r'$\sqrt{t_2^2 + \gamma^2/4}$')


# |E| against t1, periodic and open
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for t1 in t1_list:
    E_pbc = np.linalg.eigvals(nh_ssh(N_pbc, t1, t2, gamma, periodic=True))
    E_obc = np.linalg.eigvals(nh_ssh(N, t1, t2, gamma))
    axes[0].plot(np.full(2 * N_pbc, t1), np.abs(E_pbc), 'k.', ms=0.5)
    E_obc = np.sort(np.abs(E_obc))
    axes[1].plot(np.full(2 * N - 2, t1), E_obc[2:], 'k.', ms=1)
    axes[1].plot([t1, t1], E_obc[:2], 'r.', ms=1.5)
axes[0].set_title(f'periodic chain, {N_pbc} cells')
axes[1].set_title(f'open chain, {N} cells')
for ax in axes:
    add_lines(ax)
    ax.set_xlabel('$t_1 / t_2$')
    ax.set_ylim(0, 3.2)
axes[0].set_ylabel('$|E| / t_2$')
axes[1].plot([], [], 'r.', label='two states closest to $E=0$')
axes[1].legend(loc='upper left', fontsize=8)
fig.suptitle(rf'Non-Hermitian SSH, $\gamma = 4/3$')
fig.tight_layout()
fig.savefig('figures/nh_ssh_spectrum.png', dpi=200)

# check: the open chain is similar to a Hermitian SSH chain with intracell hopping
# sqrt((t1 - gamma/2)(t1 + gamma/2)). That's only real (Hermitian) for t1 > gamma/2.
for t1 in [0.8, 1.0, 1.2, 1.5, 2.0]:
    E = np.linalg.eigvals(nh_ssh(N, t1, t2, gamma))
    E_herm = np.linalg.eigvalsh(ssh(N, np.sqrt((t1 - gamma / 2) * (t1 + gamma / 2)), t2))
    error = max(np.min(np.abs(E_herm - e)) for e in E)
    print(f't1 = {t1}: max distance from Hermitian SSH spectrum = {error:.1e}')
    assert error < 1e-10
# so the zero modes exist when sqrt(t1^2 - gamma^2/4) < t2, i.e. t1 < sqrt(t2^2 + gamma^2/4)

# all eigenstates of the open chain pile up at the left edge
t1 = 1.0
r = np.sqrt(abs((t1 - gamma / 2) / (t1 + gamma / 2)))
E, psi = np.linalg.eig(nh_ssh(N, t1, t2, gamma))
psi = psi / np.linalg.norm(psi, axis=0)
weight = np.abs(psi[0::2]) ** 2 + np.abs(psi[1::2]) ** 2     # weight per unit cell
# t1 = 1 is inside the topological phase, so the two |E| closest to zero are the edge modes
# (split by ~1e-3 at N = 20, so a fixed cutoff doesn't work)
zero = np.zeros(2 * N, dtype=bool)
zero[np.argsort(np.abs(E))[:2]] = True
print(f't1 = {t1}: edge-mode energies {np.sort(np.abs(E))[:2]}, next |E| = {np.sort(np.abs(E))[2]:.3f}')
cells = np.arange(1, N + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(cells, weight[:, ~zero], color='grey', lw=0.7)
axes[0].plot(cells, weight[:, zero], color='C3', lw=1.5)
axes[0].plot([], [], color='grey', label=f'{np.sum(~zero)} bulk states')
axes[0].plot([], [], color='C3', label=f'{np.sum(zero)} zero modes')
axes[0].set_ylabel(r'$|\psi_{A,n}|^2 + |\psi_{B,n}|^2$')
axes[1].semilogy(cells, weight[:, ~zero], color='grey', lw=0.7)
axes[1].semilogy(cells, weight[:, zero], color='C3', lw=1.5)
axes[1].semilogy(cells, np.max(weight[0, ~zero]) * r ** (2 * (cells - 1)), 'k--',
                 label=r'$\propto r^{2n}$, $r = \sqrt{|t_1-\gamma/2|/|t_1+\gamma/2|}$')
axes[1].set_ylim(1e-16, 1)
for ax in axes:
    ax.set_xlabel('unit cell $n$')
    ax.legend()
axes[0].set_title(rf'open chain, all eigenstates, $t_1 = {t1}$')
axes[1].set_title('same, log scale')
fig.tight_layout()
fig.savefig('figures/nh_ssh_skin_states.png', dpi=200)

# winding number on the real Brillouin zone and with k shifted into the complex plane.
# E^2 = h_plus * h_minus, with h_plus = (t1 + gamma/2) + t2 e^(-ik) (upper-right element,
# the same one as h(k) in Part 1) and h_minus = (t1 - gamma/2) + t2 e^(ik).
# With e^(ik) -> r e^(ik), e^(-ik) -> e^(-ik)/r.
k = np.linspace(0, 2 * np.pi, 2001)
W_bloch = []
W_gbz = []
for t1 in t1_list:
    r = np.sqrt(abs((t1 - gamma / 2) / (t1 + gamma / 2)))
    for rr, W in [(1, W_bloch), (r, W_gbz)]:
        h_plus = (t1 + gamma / 2) + t2 * np.exp(-1j * k) / rr
        h_minus = (t1 - gamma / 2) + t2 * rr * np.exp(1j * k)
        # in the Hermitian case h_minus = h_plus*, so this reduces to the winding of h_plus
        W.append((winding_number(h_plus) - winding_number(h_minus)) / 2)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t1_list, W_bloch, 'C0', lw=2, label='real $k$ (Bloch)')
ax.plot(t1_list, W_gbz, 'C3', lw=2, label=r'$e^{ik} \to r e^{ik}$')
add_lines(ax)
ax.set_xlabel('$t_1 / t_2$')
ax.set_ylabel('winding number $W$')
ax.set_yticks([-1, -0.5, 0])
ax.set_title(r'Winding number, $\gamma = 4/3$')
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
fig.savefig('figures/nh_ssh_winding.png', dpi=200)

W_gbz = np.array(W_gbz)
jump = t1_list[np.argmax(np.abs(W_gbz) < 0.5)]
print(f'complex-k winding jumps from -1 to 0 at t1 = {jump:.3f} (grid step 0.005), '
      f'prediction {t1_obc:.4f}')
