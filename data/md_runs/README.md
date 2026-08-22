# Gramicidin A / DOPC Membrane MD Simulation

## System
- Peptide: Gramicidin A dimer (PDB: 1MAG, chains A/B)
- Membrane: DOPC bilayer
- Salt: KCl 2.0 M (207 K⁺, 207 Cl⁻)
- Water: TIP3P
- Force field: CHARMM36m
- Total atoms: 38,780

## Protocol
| Step | Type | Time |
|------|------|------|
| step6.0 | Energy minimization | — |
| step6.1–6.2 | NVT equilibration | 2 × 125 ps |
| step6.3–6.6 | NPT equilibration | 4 × 125–500 ps |
| step7 | NPT production | 10 ns |

## Results
| Observable | Value |
|------------|-------|
| Backbone RMSD | 1.3 ± 0.2 Å |
| Tilt angle | 9.1 ± 3.2° |
| K⁺ permeation events | 5 / 10 ns |
| Closest K⁺ to center | 8.46 Å |
| Performance | ~14 ns/day (M4 Air) |

## Software
- GROMACS 2026.2
- CHARMM-GUI Bilayer Builder
- MDAnalysis 2.10.0
- Python 3.11
