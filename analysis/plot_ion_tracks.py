import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MDAnalysis as mda

BASE = 'data/md_runs/gA_charmm-gui/charmm-gui-8342977419/gromacs'
u = mda.Universe(f'{BASE}/step7_production_ref.gro',
                 f'{BASE}/step7_production_nojump.xtc')

protein   = u.select_atoms('protein')
potassium = u.select_atoms('resname POT')
chainA    = u.select_atoms('bynum 1:250')
chainB    = u.select_atoms('bynum 251:500')

protein_z_mean = 41.69
channel_half   = 12.0
z_low  = protein_z_mean - channel_half
z_high = protein_z_mean + channel_half

n_k    = len(potassium)
n_frames = len(u.trajectory)
k_z    = np.zeros((n_frames, n_k))
times  = np.zeros(n_frames)
tilts  = np.zeros(n_frames)

for i, ts in enumerate(u.trajectory):
    k_z[i, :]  = potassium.positions[:, 2]
    times[i]   = ts.time / 1000  # ns

    com_A = chainA.center_of_mass()
    com_B = chainB.center_of_mass()
    axis  = com_A - com_B
    z_vec = np.array([0, 0, 1])
    cos_a = np.dot(axis, z_vec) / np.linalg.norm(axis)
    angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    tilts[i] = 180 - angle if angle > 90 else angle

# ионы внутри канала
inside_ions = []
min_dist    = np.zeros(n_k)
for idx in range(n_k):
    z = k_z[:, idx]
    min_dist[idx] = np.min(np.abs(z - protein_z_mean))
    if np.any((z > z_low) & (z < z_high)):
        inside_ions.append(idx)

# топ-5 ионов ближайших к центру
top5 = np.argsort(min_dist)[:5]

# ─── Figure ─────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 13))
gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

# 1. Tilt angle
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(times, tilts, color='steelblue', lw=1.2, alpha=0.8)
ax1.axhline(np.mean(tilts), color='red', ls='--', lw=1.5,
            label=f'Mean = {np.mean(tilts):.1f}° ± {np.std(tilts):.1f}°')
ax1.fill_between(times,
                 np.mean(tilts) - np.std(tilts),
                 np.mean(tilts) + np.std(tilts),
                 alpha=0.15, color='red')
ax1.set_xlabel('Time (ns)', fontsize=11)
ax1.set_ylabel('Tilt angle (°)', fontsize=11)
ax1.set_title('Gramicidin A dimer tilt angle vs membrane normal', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 2. Треки топ-5 K+ ближайших к центру
ax2 = fig.add_subplot(gs[1, :])
colors = plt.cm.tab10(np.linspace(0, 0.5, 5))

for rank, ion_idx in enumerate(top5):
    label = f'K⁺ #{ion_idx} (min dist = {min_dist[ion_idx]:.1f} Å)'
    ax2.plot(times, k_z[:, ion_idx],
             lw=1.8, alpha=0.9,
             color=colors[rank], label=label)

# все остальные серым
for idx in range(n_k):
    if idx not in top5:
        ax2.plot(times, k_z[:, idx],
                 lw=0.4, alpha=0.08, color='gray')

ax2.axhspan(z_low, z_high, alpha=0.12, color='green', label='Channel region')
ax2.axhline(protein_z_mean, color='green', ls=':', lw=1.5, label='Channel center')
ax2.set_xlabel('Time (ns)', fontsize=11)
ax2.set_ylabel('Z coordinate (Å)', fontsize=11)
ax2.set_title(f'K⁺ ion tracks — top 5 closest to channel center highlighted', fontsize=12)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.3)

# 3. Z-distribution K+
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(k_z.flatten(), bins=80,
         color='steelblue', edgecolor='white', lw=0.3)
ax3.axvspan(z_low, z_high, alpha=0.2, color='green', label='Channel region')
ax3.axvline(protein_z_mean, color='green', ls=':', lw=1.5)
ax3.set_xlabel('Z coordinate (Å)', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('K⁺ Z-distribution (all frames)', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# 4. Tilt distribution
ax4 = fig.add_subplot(gs[2, 1])
ax4.hist(tilts, bins=40,
         color='darkorange', edgecolor='white', lw=0.3)
ax4.axvline(np.mean(tilts), color='red', ls='--', lw=1.5,
            label=f'Mean = {np.mean(tilts):.1f}°')
ax4.set_xlabel('Tilt angle (°)', fontsize=11)
ax4.set_ylabel('Count', fontsize=11)
ax4.set_title('Tilt angle distribution', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.suptitle('Gramicidin A / DOPC — Ion channel geometry & K⁺ dynamics (10 ns)',
             fontsize=13, fontweight='bold')
plt.savefig('analysis/figures/ion_channel_analysis.png',
            dpi=150, bbox_inches='tight')
print("Saved: analysis/figures/ion_channel_analysis.png")
print(f"\nSummary:")
print(f"  Tilt angle:        {np.mean(tilts):.1f} ± {np.std(tilts):.1f}°")
print(f"  K+ inside channel: {len(inside_ions)} events")
print(f"  Closest K+ to center: {np.min(min_dist):.2f} Å")
