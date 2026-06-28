from Bio.PDB import PDBParser, PDBIO, Select
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(BASE_DIR, "data", "structures", "pdb1mag.ent")
output_file = os.path.join(BASE_DIR, "data", "structures", "gramicidin_clean.pdb")

parser = PDBParser(QUIET=True)
structure = parser.get_structure("gA", input_file)

class ProteinOnly(Select):
    # оставим только стандартные остатки
    def accept_residue(self,residue):
        return residue.id[0]==' '
io = PDBIO()
io.set_structure(structure)
io.save(output_file, ProteinOnly())

print("Saved cleaned structure to", output_file)