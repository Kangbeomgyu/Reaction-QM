import sys
import h5py
import matplotlib.pyplot as plt

# Draw energy profile from IRC energies
def draw_energy_profile(energy_array):

    energy_list_to_product_side = []
    energy_list_to_reactant_side = []
    TS_energy = energy_array[0]
    previous_E = TS_energy
    is_first = True
    for energy in energy_array:
        if (TS_energy - energy) < (energy - previous_E):
            is_first = False
        previous_E = energy
        if is_first:
            energy_list_to_product_side.append(energy)
        else:
            energy_list_to_reactant_side.append(energy)
    energy_list = energy_list_to_reactant_side[::-1] + energy_list_to_product_side

    plt.figure(figsize=(8,8))
    plt.tick_params(labelsize=16)
    plt.xlabel('IRC Step', fontsize=16)
    plt.ylabel('Energy (eV)', fontsize=16)
    plt.title('Energy Profile', fontsize=16)
    plt.plot(energy_list, color='darkblue', linewidth=2)
    plt.show()

# Switch the file path if you want to read another IRC data
h5file = h5py.File('B3LYPD3_TZVP_IRC_4_15.h5', 'r')
target = sys.argv[1] if len(sys.argv) > 1 else 'RXN_0000000001'

for num_atoms_key, TS_dict in h5file.items():
    for TS_name_key, TS_info_dict in TS_dict.items():

        if TS_name_key != target:
            continue
        print(f"---- {TS_name_key} ----")
        numbers = TS_info_dict['numbers'][()]
        coords = TS_info_dict['coords'][()]
        energies = TS_info_dict['energies'][()]
        forces = TS_info_dict['forces'][()]
        print(f"Atomic numbers: {numbers}")
        print(f"Coordinates: {coords}")
        print(f"Energies: {energies}")
        print(f"Forces: {forces}")
        draw_energy_profile(energies)
        exit()