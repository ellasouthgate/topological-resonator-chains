import numpy as np
import matplotlib.pyplot as plt
from chains import hatano_nelson, winding_number

t = 1.0
g = 0.3
# N = 40 keeps the open-chain eigenvalues accurate to ~1e-11 at g = 0.3 (~1e-8 at g = 0.5);
# the last figure shows eig breaking down for larger N.
N = 40

# periodic and open spectra
E_pbc = np.linalg.eigvals(hatano_nelson(N, t, g, periodic=True))
E_obc = np.linalg.eigvals(hatano_nelson(N, t, g))

k = np.linspace(0, 2 * np.pi, 801)
ellipse = 2 * t * np.cos(k) - 2j * g * np.sin(k)
m = np.arange(1, N + 1)
E_obc_exact = 2 * np.sqrt((t + g) * (t - g)) * np.cos(m * np.pi / (N + 1))

kn = 2 * np.pi * np.arange(N) / N
E_pbc_exact = 2 * t * np.cos(kn) - 2j * g * np.sin(kn)
# match each numerical eigenvalue to the nearest exact one
pbc_error = max(np.min(np.abs(E_pbc_exact - E)) for E in E_pbc)
obc_error = np.max(np.abs(np.sort(E_obc.real) - np.sort(E_obc_exact)))
print(f'periodic: max distance from 2t cos k - 2ig sin k = {pbc_error:.1e}')
print(f'open: max |Im E| = {np.max(np.abs(E_obc.imag)):.1e}, '
      f'max error vs 2 sqrt((t+g)(t-g)) cos(m pi/(N+1)) = {obc_error:.1e}')
assert pbc_error < 1e-10 and obc_error < 1e-8

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(ellipse.real, ellipse.imag, color='C0', lw=0.8, label=r'$2t\cos k - 2ig\sin k$')
ax.plot(E_pbc.real, E_pbc.imag, 'o', color='C0', ms=5, label='periodic chain')
ax.plot(E_obc.real, E_obc.imag, 's', color='C3', ms=4, label='open chain')
ax.plot(E_obc_exact, np.zeros(N), 'k+', ms=8, label=r'$2\sqrt{(t+g)(t-g)}\cos(m\pi/(N+1))$')
ax.set_xlabel('Re $E / t$')
ax.set_ylabel('Im $E / t$')
ax.set_title(f'Hatano-Nelson spectra, $g/t = {g}$, N = {N}')
ax.set_aspect('equal')
ax.set_ylim(-1.2, 1.2)
ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig('figures/hn_spectra.png', dpi=200)

# all open-chain eigenstates
# H = S H_sym S^-1 with S = diag(r^n), r = sqrt((t+g)/(t-g)), so every
# eigenstate is a standing wave times r^n and |psi|^2 grows by (t+g)/(t-g) per site.
sites = np.arange(1, N + 1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, gg in zip(axes[:2], [g, -g]):
    E, psi = np.linalg.eig(hatano_nelson(N, t, gg))
    psi = psi / np.linalg.norm(psi, axis=0)
    ax.plot(sites, np.abs(psi) ** 2, lw=0.8)
    ax.set_title(f'all {N} eigenstates, $g/t = {gg:+g}$')
    ax.set_xlabel('site $n$')
    ax.set_ylabel(r'$|\psi_n|^2$')

E, psi = np.linalg.eig(hatano_nelson(N, t, g))
psi = psi / np.linalg.norm(psi, axis=0)
ax = axes[2]
ax.semilogy(sites, np.abs(psi) ** 2, lw=0.6, color='grey')
slope_line = np.max(np.abs(psi[-1]) ** 2) * ((t + g) / (t - g)) ** (sites - N)
ax.semilogy(sites, slope_line, 'r--', lw=1.5, label=r'$\propto ((t+g)/(t-g))^n$')
ax.set_ylim(1e-14, 1)
ax.set_title(f'all {N} eigenstates,log scale, $g/t = {g:+g}$')
ax.set_xlabel('site $n$')
ax.set_ylabel(r'$|\psi_n|^2$')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig('figures/hn_skin_states.png', dpi=200)

# fitted slope of the envelope, from the largest |psi|^2 at each site
envelope = np.max(np.abs(psi) ** 2, axis=1)
slope = np.polyfit(sites[5:-5], np.log(envelope[5:-5]), 1)[0]
print(f'slope of ln|psi|^2: fitted {slope:.4f}, predicted ln((t+g)/(t-g)) = {np.log((t + g) / (t - g)):.4f}')

# winding of the periodic spectrum around E = 0, and where the states sit
g_list = np.linspace(-0.5, 0.5, 101)
winding = []
mean_position = []
for gg in g_list:
    winding.append(winding_number(2 * t * np.cos(k) - 2j * gg * np.sin(k)))
    E, psi = np.linalg.eig(hatano_nelson(N, t, gg))
    weight = np.abs(psi) ** 2 / np.sum(np.abs(psi) ** 2, axis=0)
    mean_position.append(np.mean(sites @ weight))
winding = np.array(winding)
winding[np.isclose(g_list, 0)] = np.nan   # at g = 0 the ellipse collapses onto the real axis through E = 0

fig, axes = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
axes[0].plot(g_list, winding, 'o', ms=3, label='winding of periodic spectrum around $E=0$')
axes[0].set_ylabel('winding number')
axes[0].set_yticks([-1, 0, 1])
axes[0].legend(loc='center left', fontsize=8)
axes[1].plot(g_list, mean_position, 'C1', label='open chain, averaged over eigenstates')
axes[1].axhline((N + 1) / 2, color='grey', ls='--', lw=0.8, label='middle of chain')
axes[1].set_ylabel(r'mean position $\langle n \rangle$')
axes[1].set_xlabel('$g / t$')
axes[1].legend(loc='lower right', fontsize=8)
fig.tight_layout()
fig.savefig('figures/hn_winding.png', dpi=200)

# how far eig drifts from the exact open-chain spectrum as N grows.
# Rounding errors get amplified by the condition number of S = diag(r^n), which is r^(N-1).
N_list = np.arange(10, 161, 5)
fig, ax = plt.subplots(figsize=(6, 4.5))
for gg in [0.1, 0.3, 0.5]:
    error = []
    for NN in N_list:
        E = np.linalg.eigvals(hatano_nelson(NN, t, gg))
        mm = np.arange(1, NN + 1)
        E_exact = 2 * np.sqrt((t + gg) * (t - gg)) * np.cos(mm * np.pi / (NN + 1))
        error.append(max(np.min(np.abs(E_exact - e)) for e in E))
    ax.semilogy(N_list, error, 'o-', ms=3, label=f'$g/t = {gg}$')
    print(f'g = {gg}: max eigenvalue error at N = {N}: {error[list(N_list).index(N)]:.1e}')
ax.axvline(N, color='grey', ls='--', lw=0.8, label=f'N = {N} used above')
ax.set_xlabel('chain length $N$')
ax.set_ylabel('max distance from exact eigenvalue')
ax.set_title('Open Hatano-Nelson chain: accuracy of np.linalg.eig')
ax.legend()
fig.tight_layout()
fig.savefig('figures/hn_eig_accuracy.png', dpi=200)
