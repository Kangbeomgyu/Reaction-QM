import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from matplotlib.colors import LogNorm

def draw_adjacency_changes(ax, our_adj_change_types):
    sorted_dict = dict(sorted(our_adj_change_types.items()))
    rxn_types_unique, rxn_counts = sorted_dict.keys(), sorted_dict.values()
    rxn_types_str = [str(t) for t in rxn_types_unique]

    bar1 = ax.bar(rxn_types_str, rxn_counts, width=0.15, alpha=alpha, label='Our')
    ax.bar_label(bar1, padding=1, fmt='%d', fontsize=8)
    ax.set_yscale('log')
    ax.margins(y=0.1)
    ax.set_ylim(bottom=10)
    ax.set_xticks(range(len(rxn_types_str)))
    ax.set_xticklabels(rxn_types_str)
    ax.tick_params(which='both', labelsize=12)
    ax.set_xlabel('Adjacency Change Types', fontsize=axis_fontsize, fontweight='bold')
    ax.set_ylabel('N$_{reactions}$', fontsize=axis_fontsize, fontweight='bold')
    ax.text(0.01, 0.99, 'a', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

def draw_bond_order_changes(ax, our_bo_change_types):
    sorted_dict = dict(sorted(our_bo_change_types.items()))
    bo_unique, bo_counts = sorted_dict.keys(), sorted_dict.values()
    bo_unique_str = [str(t) for t in bo_unique]

    bar1 = ax.bar(bo_unique_str, bo_counts, width=0.15, alpha=alpha, label='Our')
    ax.margins(y=0.1)
    ax.bar_label(bar1, padding=1, fmt='%d', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(bottom=10)
    ax.set_xticks(range(len(bo_unique_str)))
    ax.set_xticklabels(bo_unique_str)
    ax.tick_params(which='both', labelsize=12)
    ax.set_xlabel('Bond Order Change Types', fontsize=axis_fontsize, fontweight='bold')
    ax.set_ylabel('N$_{reactions}$', fontsize=axis_fontsize, fontweight='bold')
    ax.text(0.01, 0.99, 'b', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

def draw_rp_types(ax, our_rp_types):
    sorted_dict = dict(sorted(our_rp_types.items()))
    rp_types_unique, rp_counts = sorted_dict.keys(), sorted_dict.values()
    rp_types_str = [str(t) for t in rp_types_unique]

    bar1 = ax.bar(rp_types_str, rp_counts, width=0.15, alpha=alpha, label='Our')
    ax.margins(y=0.1)
    ax.bar_label(bar1, padding=1, fmt='%d', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(bottom=10)
    ax.set_xticks(range(len(rp_types_str)))
    ax.set_xticklabels(rp_types_str)
    ax.tick_params(which='both', labelsize=12)
    ax.set_xlabel('Reactant-Product Pair Types', fontsize=axis_fontsize, fontweight='bold')
    ax.set_ylabel('N$_{reactions}$', fontsize=axis_fontsize, fontweight='bold')
    ax.text(0.01, 0.99, 'c', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

def draw_num_heavys(ax, our_num_heavys):
    sorted_dict = dict(sorted(our_num_heavys.items()))
    heavy_unique, dft_heavy_counts = sorted_dict.keys(), sorted_dict.values()
    heavy_unique_str = ['4≥', '5', '6', '7', '8', '9', '10']

    bar1 = ax.bar(heavy_unique_str, dft_heavy_counts, width=0.15, alpha=alpha, label='Our')
    ax.margins(y=0.1)
    ax.bar_label(bar1, padding=1, fmt='%d', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(bottom=10)
    ax.set_xticks(range(len(heavy_unique_str)))
    ax.set_xticklabels(heavy_unique_str, ha='center')
    ax.tick_params(which='both', labelsize=12)
    ax.set_xlabel('Number of Heavy Atoms', fontsize=axis_fontsize, fontweight='bold')
    ax.set_ylabel('N$_{reactions}$', fontsize=axis_fontsize, fontweight='bold')
    ax.text(0.01, 0.99, 'd', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left')

def draw_bond_changes(ax, our_bond_changes, draw_colorbar=True, maximum=None):
    elements = ['C', 'H', 'O', 'N', 'S', 'P', 'Cl', 'F', 'Si']
    # Build a square matrix of counts
    idx = {el:i for i,el in enumerate(elements)}
    mat = np.zeros((len(elements), len(elements)), dtype=np.int64)
    for (a,b), n in our_bond_changes.items():
        i, j = idx[a], idx[b]
        mat[i,j] += n
        if i != j:
            mat[j,i] += n  # symmetric

    if maximum is not None:
        norm_max = maximum
    else:
        norm_max = mat.max()
    im = ax.imshow(mat, interpolation="nearest", aspect="equal", norm=LogNorm(vmin=100, vmax=norm_max))
    ax.set_xticks(range(len(elements)))
    ax.set_yticks(range(len(elements)))
    ax.set_xticklabels(elements)
    ax.set_yticklabels(elements)
    ax.tick_params(which='both', labelsize=12)
    ax.invert_yaxis()
    if draw_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Adjacency Change Frequency", fontsize=axis_fontsize, fontweight='bold')
    else:
        ax.text(0.01, 0.99, 'e', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')
    return norm_max

def aggregate_data(rxn_infos):
    adj_change_types = defaultdict(int)
    rp_types = defaultdict(int)
    num_heavys = defaultdict(int)
    bond_changes = defaultdict(int)
    bo_change_types = defaultdict(int)
    
    # Aggregate data
    for rxn_info in rxn_infos:
        rxn_smiles, adj_change, rp_type, num_heavy, bond_change_list, bo_change = rxn_info
        # Filter out invalid reactions
        if adj_change == (0,0):
            continue
        # Adjacency change types
        if max(adj_change) == 1:
            adj_change_types['b1'] += 1
        elif max(adj_change) == 2:
            adj_change_types['b2'] += 1
        elif max(adj_change) == 3:
            adj_change_types['b3'] += 1
        elif max(adj_change) == 4:
            adj_change_types['b4'] += 1
        elif max(adj_change) == 5:
            adj_change_types['b5'] += 1
        else:
            adj_change_types['b>5'] += 1
        # RP types
        if rp_type == (1,1):
            rp_types['R1P1'] += 1
        elif rp_type == (1,2):
            rp_types['R1P2'] += 1
        elif rp_type[0] == 1:
            rp_types['R1P≥3'] += 1
        elif rp_type == (2,2):
            rp_types['R2P2'] += 1
        elif rp_type == (2,3):
            rp_types['R2P3'] += 1
        elif rp_type[0] == 2:
            rp_types['R2P≥4'] += 1
        elif rp_type[0] > 2:
            rp_types['R≥3P≥3'] += 1
        # Number of heavy atoms
        if num_heavy <= 4:
            num_heavys[4] += 1
        else:
            num_heavys[num_heavy] += 1
        # Bond changes
        for bond_change in bond_change_list:
            bond_changes[bond_change] += 1
        # Bond order change types
        if max(bo_change) == 1:
            bo_change_types['b1'] += 1
        elif max(bo_change) == 2:
            bo_change_types['b2'] += 1
        elif max(bo_change) == 3:
            bo_change_types['b3'] += 1
        elif max(bo_change) == 4:
            bo_change_types['b4'] += 1
        elif max(bo_change) == 5:
            bo_change_types['b5'] += 1
        else:
            bo_change_types['b>5'] += 1

    return (adj_change_types, rp_types, num_heavys, bond_changes, bo_change_types)


if '__main__' == __name__:

    import sys
    # load in data
    try:
        dft_statistics_file = sys.argv[1]
    except:
        print ('File directory is not provided !!!')
        print ('Ex: reaction_statistics_dft.pkl')

    with open(dft_statistics_file, 'rb') as f:
        rxn_info_dict_dft = pickle.load(f)

    # Aggregate data only for DFT results because of enormous xtb dataset size
    dft_rxn_infos = rxn_info_dict_dft.values()
    dft_adj_change_types, dft_rp_types, dft_num_heavys, dft_bond_changes, dft_bo_change_types = aggregate_data(dft_rxn_infos)
    print('Done with DFT data aggregation.')

    # Create a figure with 6 subplots
    fig = plt.figure(figsize=(18, 13))

    ax1 = plt.subplot2grid(shape=(3, 12), loc=(0, 0), colspan=6) # adjacency changes
    ax2 = plt.subplot2grid(shape=(3, 12), loc=(0, 6), colspan=6) # bond order changes
    ax3 = plt.subplot2grid(shape=(3, 12), loc=(1, 0), colspan=6) # rp types
    ax4 = plt.subplot2grid(shape=(3, 12), loc=(1, 6), colspan=6) # num heavys
    ax5 = plt.subplot2grid(shape=(3, 12), loc=(2, 0), colspan=4) # bond change heatmap

    # Miscellaneous settings
    alpha = 0.5
    title_fontsize = 14
    axis_fontsize = 14
    tick_fontsize = 12
    bar_fontsize = 9

    draw_adjacency_changes(ax1, dft_adj_change_types)
    draw_bond_order_changes(ax2, dft_bo_change_types)
    draw_rp_types(ax3, dft_rp_types)
    draw_num_heavys(ax4, dft_num_heavys)
    max = draw_bond_changes(ax5, dft_bond_changes, draw_colorbar=True, maximum=None)

    plt.tight_layout()
    plt.savefig('reaction_statistics.png', dpi=300)
    plt.show()
