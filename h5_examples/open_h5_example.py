import sys
import h5py

# Switch the file path into 'GFN2_xTB.h5' if you want to read GFN2 data
h5file = h5py.File('B3LYP_TZVP.h5', 'r')
target = sys.argv[1] if len(sys.argv) > 1 else 'RXN_0000000001'
if target not in h5file:
    raise ValueError(f"Target reaction '{target}' not found in the HDF5 file.")

for rxn_name_key, reaction_point_dict in h5file.items():
    if rxn_name_key != target:
        continue
    print(f"---- Information of {rxn_name_key} ----")

    reactants_data_dict = reaction_point_dict.get('R')
    products_data_dict = reaction_point_dict.get('P')
    ts_data_dict = reaction_point_dict.get('TS')

    # Parse data of reactants
    for idx, (species_key, species_data) in enumerate(reactants_data_dict.items()):
        print(f"---- Reactant {idx+1}: {species_key} ----")
        print(f"E, H, G : {species_data['EHG'][()]}")
        print('xyz coordinates:')
        for atom_num, coord in zip(species_data['atomic_number'][()], species_data['coords'][()]):
            print(f"{atom_num} {coord[0]} {coord[1]} {coord[2]}")

    # Parse data of products
    for idx, (species_key, species_data) in enumerate(products_data_dict.items()):
        print(f"---- Product {idx+1}: {species_key} ----")
        print(f"E, H, G : {species_data['EHG'][()]}")
        print('xyz coordinates:')
        for atom_num, coord in zip(species_data['atomic_number'][()], species_data['coords'][()]):
            print(f"{atom_num} {coord[0]} {coord[1]} {coord[2]}")

    # Parse data of transition state
    print("---- Transition State ----")
    print(f"E, H, G : {ts_data_dict['EHG'][()]}")
    print('xyz coordinates:')
    for atom_num, coord in zip(ts_data_dict['atomic_number'][()], ts_data_dict['coords'][()]):
        print(f"{atom_num} {coord[0]} {coord[1]} {coord[2]}")

    break  # Print only one reaction