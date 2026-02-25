import pickle
import numpy as np
import matplotlib.pyplot as plt


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
    name='force_histogram'
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
    fig.suptitle(suptitle, fontsize=18, y=0.98)

    # Ticks & layout
    for ax in axes:
        if ax.has_data():
            ax.tick_params(axis='x', which='both', labelsize=14)
            ax.tick_params(axis='y', which='both', labelsize=12)

    plt.tick_params(labelsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{name}.png", dpi=300)
    plt.show()


if __name__ == '__main__':

    # Load sampled force data
    try:
        file_name = sys.argv[1]
    except:
        print ('File directory is not provided !!!')
        print ('Ex: element_force_dict_sampled.pkl')

    elements_force_dict = {}
    with open(file_name, 'rb') as f:
        e_f_dict = pickle.load(f)
    for element, forces in e_f_dict.items():
        np_forces = np.asarray(forces, dtype=np.float32)
        norm_force = np.linalg.norm(np_forces, axis=1)  # convert to magnitudes
        elements_force_dict[element] = norm_force

    # Print force frequencies
    for element, forcelist in elements_force_dict.items():
        print(f'{element} : {len(forcelist)}')

    # Plot
    plot_force_histograms_grid(elements_force_dict, value_range=(0, 10), bins=20, use_abs=True)
