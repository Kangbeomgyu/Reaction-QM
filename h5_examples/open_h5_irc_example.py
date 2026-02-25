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
file_name = sys.argv[1]
h5file = h5py.File(file_name, 'r')
target = sys.argv[2] if len(sys.argv) > 2 else 'RXN_0000000001'

# Check the format of the data ...
length = 10
if target.isdigit():
    rxn_number = target.rjust(length,'0')
    target = f'RXN_{rxn_number}'

TS_info_dict = h5file[target]
print(f"---- {target} ----")
numbers = TS_info_dict['atomic_numbers'][()]
coords = TS_info_dict['coordinates'][()]
energies = TS_info_dict['energies'][()]
forces = TS_info_dict['forces'][()]
print(f"Atomic numbers: {numbers}")
print(f"Coordinates: {coords}")
print(f"Energies: {energies}")
print(f"Forces: {forces}")

# Activate this to see IRC trajectory ...
#draw_energy_profile(energies)
