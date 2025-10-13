import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import defaultdict
import pickle

def pair_distance_heatmap_from_dict(pair_to_distances, distance_bins=None,
                                    normalize=False, log=False, ax=None,
                                    bin_width=0.05, cmap='viridis', dmin=0.8, dmax=2.0):
    """
    pair_to_distances: dict { 'C-H': array_like_of_distances, ... }
    """
    uniq_pairs = np.array(sorted(pair_to_distances.keys()))
    n_pairs = uniq_pairs.size

    if distance_bins is None:
        nbins = int(np.round((dmax - dmin) / bin_width))
        upper = np.nextafter(dmax, np.inf)
        distance_bins = np.linspace(dmin, upper, nbins + 1)
        distance_bins = np.asarray(distance_bins, dtype=float)

    # Fill counts per pair without expanding labels
    H = np.zeros((n_pairs, nbins), dtype=np.float64)
    for i, p in enumerate(uniq_pairs):
        arr = np.asarray(pair_to_distances[p], dtype=float)
        if arr.size == 0:
            continue
        hist, _ = np.histogram(arr, bins=distance_bins)
        H[i] = hist

    if normalize:
        col_sums = H.sum(axis=1, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        H = H / col_sums

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    else:
        fig = ax.figure

    extent = [-0.5, n_pairs - 0.5, dmin, dmax]
    im = ax.imshow(H.T, origin='lower', aspect='auto', extent=extent,
                   cmap=cmap, norm=LogNorm(vmin=1, vmax=H.max()) if log else None)

    ax.set_xlabel('Atom-atom pair', fontsize=14)
    ax.set_ylabel('Relative distance (r/r$_0$)', fontsize=14)
    ax.set_ylim((dmin, dmax))
    ax.set_xticks(np.arange(n_pairs))
    ax.set_xticklabels(uniq_pairs, rotation=45, ha='right')
    ax.tick_params(labelsize=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Frequency', fontsize=14)

    ax.set_title('Distance frequency by element pair', fontsize=18)
    fig.tight_layout()
    fig.savefig('final_distance_heatmap.png', dpi=300)
    plt.show()


if __name__ == '__main__':

    # Load sampled interatomic distance data
    file_name = 'element_distance_sampled.npy'
    np_arr = np.load(file_name)

    # Run get_r_zeros.py first to generate r_zero_dict.pkl
    with open('r_zero_dict.pkl', 'rb') as f:
        r_zero_dict = pickle.load(f)

    pair_distance_dict = defaultdict(list)
    num2symbol = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 14:'Si', 15:'P', 16:'S', 17:'Cl'}
    for info in np_arr:
        atom1, atom2, distance = info
        atom_tuple_key = tuple(sorted([num2symbol[atom1], num2symbol[atom2]]))
        r_zero = r_zero_dict[atom_tuple_key]
        relative_r = distance/r_zero
        if relative_r < 0.8 or relative_r > 2.0:
            continue
        atom_dash_key = '-'.join(sorted([num2symbol[atom1], num2symbol[atom2]]))
        pair_distance_dict[atom_dash_key].append(relative_r)
    
    for atom_key, dist_list in pair_distance_dict.items():
        print(f'Distance data of {atom_key} : {len(dist_list)}')
        pair_distance_dict[atom_key] = np.asarray(dist_list)

    pair_distance_heatmap_from_dict(pair_distance_dict, bin_width=0.05, log=True)
