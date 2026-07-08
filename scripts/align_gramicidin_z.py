"""
Выравнивание gramicidin A по оси Z.
Ось канала уже почти Z (0.9998) — просто центрируем и сохраняем.
"""
from Bio.PDB import PDBParser, PDBIO
from pathlib import Path
import numpy as np

INPUT  = Path("data/md_test/gramicidin_charmm.pdb")
OUTPUT = Path("data/md_test/gramicidin_aligned.pdb")

parser = PDBParser(QUIET=True)
structure = parser.get_structure("gA", INPUT)
model = structure[0]

# собираем все координаты
all_atoms = list(model.get_atoms())
all_coords = np.array([a.get_vector().get_array() for a in all_atoms])

# центр масс
com = all_coords.mean(axis=0)
print(f"Центр масс: {com.round(3)}")

# центры цепей
coords_A = np.array([a.get_vector().get_array()
                     for a in model["A"].get_atoms()])
coords_B = np.array([a.get_vector().get_array()
                     for a in model["B"].get_atoms()])
com_A = coords_A.mean(axis=0)
com_B = coords_B.mean(axis=0)

# ось канала
axis = com_A - com_B
axis_norm = axis / np.linalg.norm(axis)
print(f"Ось канала: {axis_norm.round(4)}")
print(f"Z-компонента: {abs(axis_norm[2]):.4f}  (уже ~1.0 — только центрируем)")

# матрица вращения channel_axis → Z
z = np.array([0.0, 0.0, 1.0])
v = np.cross(axis_norm, z)
s = np.linalg.norm(v)
c = np.dot(axis_norm, z)

if s < 1e-6:
    R = np.eye(3)
else:
    vx = np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)

# применяем: coord_new = R @ (coord - com)
# BioPython: atom.coord — это np.array, можно менять напрямую
for atom in model.get_atoms():
    old = atom.get_vector().get_array()
    new = R @ (old - com)
    atom.coord = new.astype(np.float32)

# проверка после
coords_A_new = np.array([a.coord for a in model["A"].get_atoms()])
coords_B_new = np.array([a.coord for a in model["B"].get_atoms()])
com_A_new = coords_A_new.mean(axis=0)
com_B_new = coords_B_new.mean(axis=0)
axis_new = com_A_new - com_B_new
axis_new /= np.linalg.norm(axis_new)

print(f"\nПосле выравнивания:")
print(f"  com_A z={com_A_new[2]:.2f} Å")
print(f"  com_B z={com_B_new[2]:.2f} Å")
print(f"  ось канала: {axis_new.round(4)}")
print(f"  Z-компонента: {abs(axis_new[2]):.6f}")
print(f"  длина канала по Z: {abs(com_A_new[2]-com_B_new[2]):.1f} Å")

# сохраняем
io = PDBIO()
io.set_structure(structure)
io.save(str(OUTPUT))

# HETATM → ATOM
with open(OUTPUT) as f:
    lines = f.readlines()
fixed = ["ATOM  " + l[6:] if l.startswith("HETATM") else l for l in lines]
with open(OUTPUT, "w") as f:
    f.writelines(fixed)

atom_count = sum(1 for l in fixed if l.startswith("ATOM"))
print(f"\nСохранено: {OUTPUT}")
print(f"ATOM строк: {atom_count}")
