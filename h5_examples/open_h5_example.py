import sys
import h5py

def print_data(species_data):
    smiles = species_data['smiles'][()]
    EHG = species_data['EHG'][()]
    chg = species_data['charge'][()]
    multiplicity = species_data['multiplicity'][()]
    z_list = species_data['atomic_numbers'][()]
    coords = species_data['coordinates'][()]
    print(f"SMILES: {smiles}")
    print(f"E, H, G (Hartree): {EHG}")
    print(f"Charge : {chg}")
    print(f"Spin multiplicity : {multiplicity}")
    print('xyz coordinates (Å):')
    for atom_num, coord in zip(z_list, coords):
        print(f"{atom_num} {coord[0]} {coord[1]} {coord[2]}")

# Enter the HDF5 file path and target reaction id (optional)
file_name = sys.argv[1]
h5file = h5py.File(file_name, 'r')
target = sys.argv[2] if len(sys.argv) > 2 else 'RXN_0000000001'

# Check the format of the data ...
length = 10
if target.isdigit():
    rxn_number = target.rjust(length,'0')
    target = f'RXN_{rxn_number}'

found = False
for root_key, root_value in h5file.items(): # root_key: RXN_A to RXN_B, root_value: dict of molecule + ts data
    rxn_name_keys = root_value.keys() 
    print (f'Checking files in {root_key} ...')
    if target not in rxn_name_keys:
        continue

    print ('Found desired key !')
    found = True
    molecules_and_ts_dict = root_value[target] 
    print(f"---- Information of {target} ----")

    # Parse data of reactants, products, and transition state
    for molecule_tag, molecule_data in molecules_and_ts_dict.items():
        print(f"-------------------- {molecule_tag} --------------------")
        print_data(molecule_data)

    break  # Print only one reaction

if not found:
    print (f'Desired reaction (={rxn_name}) not found !!')
    print ('Check the key again ...')


