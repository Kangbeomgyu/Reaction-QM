import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

name_pairs = [
"H-H", "H-C", "H-O", "H-N", "H-S", "H-P", "H-F", "H-Cl", "H-Si",
"C-C", "C-O", "C-N", "C-S", "C-P", "C-F", "C-Cl", "C-Si",
"O-O", "O-N", "O-S", "O-P", "O-F", "O-Cl", "O-Si",
"N-N", "N-S", "N-P", "N-F", "N-Cl", "N-Si",
"S-S", "S-P", "S-F", "S-Cl", "S-Si",
"P-P", "P-F", "P-Cl", "P-Si",
"F-F", "F-Cl", "F-Si",
"Cl-Cl", "Cl-Si",
"Si-Si"
]
name_tuple_list = []
for name_pair in name_pairs:
    first, second = name_pair.split('-')
    name_tuple = tuple(sorted([first, second]))
    name_tuple_list.append(name_tuple)

pairs = [
"[H][H]", "[H]C", "[H]O", "[H]N", "[H]S", "[H]P", "[H]F", "[H]Cl", "[H][SiH3]",
"CC", "CO", "CN", "CS", "CP", "CF", "CCl", "C[SiH3]",
"OO", "ON", "OS", "OP", "OF", "OCl", "O[SiH3]",
"NN", "NS", "NP", "NF", "NCl", "N[SiH3]",
"SS", "SP", "SF", "SCl", "S[SiH3]",
"PP", "PF", "PCl", "P[SiH3]",
"FF", "FCl", "F[SiH3]",
"ClCl", "Cl[SiH3]",
"[SiH3][SiH3]"
]

name_distance_dict = dict()
for smiles, name_tuple in zip(pairs, name_tuple_list):
    # Load molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens (important for 3D embedding)
    mol = Chem.AddHs(mol)

    # Generate 3D conformer
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    # Get conformer
    conf = mol.GetConformer()

    # Measure distance
    atom1 = mol.GetAtomWithIdx(0).GetSymbol()
    atom2 = mol.GetAtomWithIdx(1).GetSymbol()
    distance = rdMolTransforms.GetBondLength(conf, 0, 1)
    print(f"Distance between atom {atom1} and {atom2}: {distance:.3f} Å")
    name_distance_dict[name_tuple] = distance

with open('r_zero_dict.pkl', 'wb') as f:
    pickle.dump(name_distance_dict, f)