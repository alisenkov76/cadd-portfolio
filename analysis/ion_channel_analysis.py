import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MDAnalysis as mda
from MDAnalysis.analysis import align

# ─── Загрузка ───────────────────────────────────────────────
BASE = '/Users/anya/cadd-portfolio/data/md_runs/gA_charmm-gui/charmm-gui-8342977419/gromacs'
u = mda.Universe(f'{BASE}/step7_production_ref.gro',
                 f'{BASE}/step7_production_nojump.xtc')

print(f"Атомов: {len(u.atoms)}")
print(f"Фреймов: {len(u.trajectory)}")
print(f"Время: {u.trajectory[0].time:.1f} — {u.trajectory[-1].time:.1f} ps")

# ─── Группы ─────────────────────────────────────────────────
chainA = u.select_atoms("bynum 1:250")
chainB = u.select_atoms("bynum 251:500")
protein = u.select_atoms('protein')
potassium = u.select_atoms('resname POT')

print(f"\nChain A: {len(chainA)} атомов")
print(f"Chain B: {len(chainB)} атомов")
print(f"K+ ионов: {len(potassium)}")

# ─── Tilt angle ──────────────────────────────────────────────
times = []
tilts = []

for ts in u.trajectory:
    # вектор от COM chain B до COM chain A = ось димера
    com_A = chainA.center_of_mass()
    com_B = chainB.center_of_mass()
    axis = com_A - com_B
    
    # угол с осью Z (0,0,1)
    z = np.array([0, 0, 1])
    cos_angle = np.dot(axis, z) / (np.linalg.norm(axis) * np.linalg.norm(z))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    # приводим к диапазону 0–90°
    if angle > 90:
        angle = 180 - angle
    
    times.append(ts.time / 1000)  # ps → ns
    tilts.append(angle)

times = np.array(times)
tilts = np.array(tilts)

print(f"\nTilt angle:")
print(f"  Mean: {np.mean(tilts):.2f}°")
print(f"  Std:  {np.std(tilts):.2f}°")
print(f"  Min:  {np.min(tilts):.2f}°")
print(f"  Max:  {np.max(tilts):.2f}°")

# ─── Треки K+ вдоль Z ────────────────────────────────────────
# собираем Z-координату каждого K+ по времени
n_k = len(potassium)
k_z_tracks = np.zeros((len(u.trajectory), n_k))

# границы мембраны по Z (из density_DOPC.xvg — примерно)
# уточним через COM белка
protein_z_positions = []

for i, ts in enumerate(u.trajectory):
    k_z_tracks[i, :] = potassium.positions[:, 2]  # Z в Å
    protein_z_positions.append(protein.center_of_mass()[2])

protein_z_mean = np.mean(protein_z_positions)
print(f"\nБелок COM Z (среднее): {protein_z_mean:.2f} Å")

# ─── Ионы которые прошли через канал ─────────────────────────
# канал ~ ±15 Å от COM белка по Z
channel_half = 15.0  # Å
z_low  = protein_z_mean - channel_half
z_high = protein_z_mean + channel_half

crossing_ions = []
for ion_idx in range(n_k):
    z_traj = k_z_tracks[:, ion_idx]
    entered_low  = np.any(z_traj < z_low)
    entered_high = np.any(z_traj > z_high)
    # пересечение = побывал с обеих сторон
    if entered_low and entered_high:
        crossing_ions.append(ion_idx)

print(f"K+ ионов пересекли канал: {len(crossing_ions)} из {n_k}")

# ─── Графики ─────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

# 1. Tilt angle
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(times, tilts, color='steelblue', lw=1.2, alpha=0.8)
ax1.axhline(np.mean(tilts), color='red', ls='--', lw=1.5,
            label=f'Mean = {np.mean(tilts):.1f}°')
ax1.fill_between(times,
                 np.mean(tilts) - np.std(tilts),
                 np.mean(tilts) + np.std(tilts),
                 alpha=0.2, color='red')
ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('Tilt angle (°)')
ax1.set_title('Gramicidin A dimer tilt angle relative to membrane normal (Z)')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Треки K+ вдоль Z (все ионы)
ax2 = fig.add_subplot(gs[1, :])
for ion_idx in range(n_k):
    alpha = 0.6 if ion_idx in crossing_ions else 0.1
    color = 'red' if ion_idx in crossing_ions else 'gray'
    lw = 1.5 if ion_idx in crossing_ions else 0.5
    ax2.plot(times, k_z_tracks[:, ion_idx], 
             alpha=alpha, color=color, lw=lw)

ax2.axhline(z_low,  color='orange', ls='--', lw=1.5, label='Channel boundary')
ax2.axhline(z_high, color='orange', ls='--', lw=1.5)
ax2.axhline(protein_z_mean, color='green', ls=':', lw=1.5, label='Protein COM')
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Z coordinate (Å)')
ax2.set_title(f'K⁺ ion tracks along Z — red = crossed channel ({len(crossing_ions)} ions)')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Распределение K+ по Z
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(k_z_tracks.flatten(), bins=80, color='steelblue',
         edgecolor='white', linewidth=0.3)
ax3.axvline(z_low,  color='orange', ls='--', lw=1.5)
ax3.axvline(z_high, color='orange', ls='--', lw=1.5, label='Channel region')
ax3.axvline(protein_z_mean, color='green', ls=':', lw=1.5, label='Protein COM')
ax3.set_xlabel('Z coordinate (Å)')
ax3.set_ylabel('Count')
ax3.set_title('K⁺ Z-distribution (all frames)')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Tilt histogram
ax4 = fig.add_subplot(gs[2, 1])
ax4.hist(tilts, bins=40, color='darkorange',
         edgecolor='white', linewidth=0.3)
ax4.axvline(np.mean(tilts), color='red', ls='--', lw=1.5,
            label=f'Mean = {np.mean(tilts):.1f}°')
ax4.set_xlabel('Tilt angle (°)')
ax4.set_ylabel('Count')
ax4.set_title('Tilt angle distribution')
ax4.legend()
ax4.grid(alpha=0.3)

plt.suptitle('Gramicidin A / DOPC — Ion channel analysis (10 ns)',
             fontsize=13, fontweight='bold')
plt.savefig('/Users/anya/cadd-portfolio/analysis/figures/ion_channel_analysis.png',
            dpi=150, bbox_inches='tight')
print("\nSaved: analysis/figures/ion_channel_analysis.png")
