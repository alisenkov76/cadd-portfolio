import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def read_xvg(fname):
    x, y = [], []
    with open(fname) as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            cols = line.split()
            if len(cols) >= 2:
                x.append(float(cols[0]))
                y.append(float(cols[1]))
    return np.array(x), np.array(y)

BASE = 'data/md_runs/gA_charmm-gui/charmm-gui-8342977419/gromacs'

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# --- RMSD ---
ax1 = fig.add_subplot(gs[0, :])
t, rmsd = read_xvg(f'{BASE}/rmsd_backbone.xvg')
ax1.plot(t, rmsd * 10, color='steelblue', lw=1.5)
ax1.axhline(np.mean(rmsd * 10), color='red', ls='--', lw=1,
            label=f'Mean = {np.mean(rmsd*10):.2f} Å')
ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('RMSD (Å)')
ax1.set_title('Gramicidin A Backbone RMSD — 10 ns production')
ax1.legend()
ax1.grid(alpha=0.3)

# --- Density DOPC ---
ax2 = fig.add_subplot(gs[1, 0])
z_dopc, d_dopc = read_xvg(f'{BASE}/density_DOPC.xvg')
ax2.plot(z_dopc, d_dopc, color='darkorange', lw=1.5, label='DOPC')
ax2.set_xlabel('Z (nm)')
ax2.set_ylabel('Density (kg/m³)')
ax2.set_title('DOPC density along Z')
ax2.legend()
ax2.grid(alpha=0.3)

# --- Density Water + K+ ---
ax3 = fig.add_subplot(gs[1, 1])
z_w, d_w = read_xvg(f'{BASE}/density_water.xvg')
z_k, d_k = read_xvg(f'{BASE}/density_K.xvg')
ax3.plot(z_w, d_w, color='dodgerblue', lw=1.5, label='TIP3 water')
ax3_twin = ax3.twinx()
ax3_twin.plot(z_k, d_k, color='green', lw=1.5, label='K⁺')
ax3.set_xlabel('Z (nm)')
ax3.set_ylabel('Water density (kg/m³)', color='dodgerblue')
ax3_twin.set_ylabel('K⁺ density (kg/m³)', color='green')
ax3.set_title('Water & K⁺ density along Z')
ax3.grid(alpha=0.3)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.suptitle('Gramicidin A / DOPC membrane — MD analysis',
             fontsize=13, fontweight='bold')
plt.savefig('analysis/figures/md_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: analysis/figures/md_analysis.png")
