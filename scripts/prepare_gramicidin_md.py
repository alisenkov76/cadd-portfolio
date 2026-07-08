"""
Подготовка gramicidin A димера для MD симуляции.
Все остатки записываются как ATOM (не HETATM).
"""
from Bio.PDB import PDBParser, PDBIO, Select
from pathlib import Path

INPUT       = Path("data/structures/pdb1mag.ent")
OUT_CHARMM  = Path("data/md_test/gramicidin_charmm.pdb")
OUT_CHARMM.parent.mkdir(exist_ok=True)

RENAME = {"DLE": "LEU", "DVA": "VAL"}
REMOVE = {"FVA", "ETA"}

parser = PDBParser(QUIET=True)
structure = parser.get_structure("1MAG", INPUT)
model = structure[0]

for chain in model:
    to_remove = []
    for res in chain:
        resname = res.resname.strip()
        if resname in REMOVE:
            to_remove.append(res.get_id())
        elif resname in RENAME:
            res.resname = RENAME[resname]
            # КРИТИЧНО: меняем het-флаг на ' ' чтобы записалось как ATOM
            old_id = res.get_id()
            new_id = (' ', old_id[1], old_id[2])
            res.id = new_id
    for res_id in to_remove:
        chain.detach_child(res_id)

# принудительно пишем все как ATOM
class ForceATOM(Select):
    def accept_residue(self, residue):
        return True

    def accept_atom(self, atom):
        # сбрасываем флаг HETATM
        atom.full_id  # просто обращение
        return True

# сохраняем через прямую запись строк
# (PDBIO уважает res.id[0] как флаг HETATM/ATOM)
io = PDBIO()
io.set_structure(structure)
io.save(str(OUT_CHARMM))

# читаем обратно и проверяем
with open(OUT_CHARMM) as f:
    lines = f.readlines()

# заменяем HETATM на ATOM
fixed = []
for line in lines:
    if line.startswith("HETATM"):
        line = "ATOM  " + line[6:]
    fixed.append(line)

with open(OUT_CHARMM, "w") as f:
    f.writelines(fixed)

print(f"Сохранено: {OUT_CHARMM}")

# проверка
atom_lines  = [l for l in fixed if l.startswith("ATOM")]
hetatm_lines = [l for l in fixed if l.startswith("HETATM")]
print(f"ATOM   строк: {len(atom_lines)}")
print(f"HETATM строк: {len(hetatm_lines)}  (должно быть 0)")

# финальная последовательность
print("\nФинальная последовательность:")
seen = {}
for line in atom_lines:
    chain = line[21]
    resname = line[17:20].strip()
    resseq = int(line[22:26])
    key = (chain, resseq)
    if key not in seen:
        seen[key] = resname

for chain_id in ["A", "B"]:
    residues = [(k[1], v) for k, v in seen.items() if k[0] == chain_id]
    residues.sort()
    seq = " - ".join(f"{r[1]}{r[0]}" for r in residues)
    print(f"  Chain {chain_id}: {seq}")
