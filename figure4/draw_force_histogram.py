import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

'''
def plot_force_histograms(
    force_dict,           # {element : nd array of forces}
    bins='auto',          
    value_range=None,     # e.g., (0, 10)
    density=False,         # True to compare shapes; False to compare counts
    use_abs=False,        # take absolute values before histogramming
    xscale="linear",      # "linear" or "log"
    yscale="log",      # "linear" or "log"
    alpha=0.15,           # transparency for overlap
    figsize=(20, 10),
    xlabel="Force magnitude (eV/Å)",
    title="Overlapped force distributions by element",
    legend_loc="best",
    name = 'force_histogram'
):
    """
    force_dict: { 'C': [f1, f2, ...], 'H': [...], ... }
      values may be scalars or 3D vectors; vectors will be converted to magnitudes.
    """
    plt.figure(figsize=figsize)
    plotted_any = False

    num2symbol = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 14:'Si', 15:'P', 16:'S', 17:'Cl'}
    overlap_sequence = [1, 6, 8, 7, 16, 15, 14, 17, 9]
    # Sorted keys for stable color/legend order (you can change to list(force_dict) to keep given order)
    for idx, elem in enumerate(overlap_sequence):
        data = np.array(force_dict[elem])
        if data.size == 0:
            continue
        if use_abs:
            data = np.abs(data)
        # Remove NaNs/inf
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue

        plt.hist(
            data,
            bins=bins,
            range=value_range,
            density=density,
            histtype="step",   # filled for clearer overlap
            alpha=alpha,
            label=num2symbol[elem],
            linewidth=3.0,
        )
        plotted_any = True
        print(f'Plotted {num2symbol[elem]}')

    if not plotted_any:
        raise ValueError("No valid numeric force data found in force_dict.")

    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Count")
    plt.title(title)
    plt.yscale(yscale)
    plt.xscale(xscale)
    plt.legend(title="Element", loc=legend_loc, frameon=False, fontsize='x-large')
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    plt.show()
'''

def plot_force_histograms_grid(
    force_dict,           # {atomic_number : array-like of forces}
    bins='auto',
    value_range=None,     # e.g., (0, 10)
    density=False,        # True to compare shapes; False to compare counts
    use_abs=True,        # take absolute values before histogramming
    xscale="linear",      # "linear" or "log"
    yscale="log",         # "linear" or "log"
    alpha=0.35,           # transparency for fill
    figsize=(15, 9),
    xlabel="Force magnitude (eV/Å)",
    suptitle="Force distributions by element",
    name='real_final_force_histogram_3x3'
):

    num2symbol = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 14:'Si', 15:'P', 16:'S', 17:'Cl'}
    panel_order = [1, 6, 7, 8, 9, 14, 15, 16, 17]  # fixed order for 3x3 grid

    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    any_plotted = False

    for ax, elem in zip(axes, panel_order):
        # Get data; tolerate missing keys
        arr = np.asarray(force_dict.get(elem, []))
        if arr.size == 0:
            ax.set_axis_off()
            continue
        if use_abs:
            arr = np.abs(arr)

        # Remove NaNs/inf
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.set_axis_off()
            continue

        # Plot
        ax.hist(
            arr,
            bins=bins,
            range=value_range,
            density=density,
            histtype="stepfilled",
            alpha=alpha,
            linewidth=1.0,
        )
        ax.set_title(num2symbol.get(elem, str(elem)), fontsize=15)
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_ylim(bottom=1)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        raise ValueError("No valid numeric force data found in force_dict.")

    # Common labels
    fig.supxlabel(xlabel, fontsize=16)
    fig.supylabel("Density" if density else "Frequency", fontsize=16)
    #fig.suptitle(suptitle, fontsize=18, y=0.98)

    # Ticks & layout
    for ax in axes:
        if ax.has_data():
            ax.tick_params(axis='x', which='both', labelsize=14)
            ax.tick_params(axis='y', which='both', labelsize=12)

    plt.tick_params(labelsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{name}.pdf", dpi=300)
    plt.savefig(f"{name}.png", dpi=300)
    plt.show()


if __name__ == '__main__':

    data_directory = 'our_data'
    file_list = os.listdir(data_directory)
    #file_list = ['element_force_dict_0.pkl'] # test

    elements_force_dict = defaultdict(list)
    for file in file_list:
        full_file = os.path.join(data_directory, file)
        if file.endswith('pkl'):
            with open(full_file, 'rb') as f:
                e_f_dict = pickle.load(f)
            for element, forces in e_f_dict.items():
                np_forces = np.asarray(forces, dtype=np.float32)
                elements_force_dict[element].append(np_forces)
            print(f'Processed {file}')
    for element, force_list in elements_force_dict.items():
        elements_force_dict[element] = np.concatenate(force_list)

    for element, forcelist in elements_force_dict.items():
        print(f'{element} : {len(forcelist)}')

    plot_force_histograms_grid(elements_force_dict, value_range=(0, 10), bins=20, use_abs=True)
