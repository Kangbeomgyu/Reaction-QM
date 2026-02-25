import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def read_rxn_data(csv_file):
    rxn_infos = []
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        rxn_id = row['reaction_id']
        rxn_smiles = row['reaction_smiles']
        dE_dag = row['dE_dagger']
        dE = row['dE']
        dH_dag = row['dH_dagger']
        dH = row['dH']
        dG_dag = row['dG_dagger']
        dG = row['dG']
        dE_dag = float(dE_dag)
        dE = float(dE)
        dH_dag = float(dH_dag)
        dH = float(dH)
        dG_dag = float(dG_dag)
        dG = float(dG)
        rxn_infos.append([rxn_id, rxn_smiles, dE_dag, dE, dH_dag, dH, dG_dag, dG])

    return rxn_infos

def get_rxn_infos(edge_infos):
    rxn_infos = []
    for edge_info in edge_infos:
        rxn_id, rxn_smiles, dE_dag, dE, dH_dag, dH, dG_dag, dG = edge_info
        rxn_infos.append([rxn_id, rxn_smiles, dH, dE_dag])
        
        # Also consider reverse reaction
        dH_reverse = -dH
        dE_dag_reverse = dE_dag - dE
        r_smiles, p_smiles = rxn_smiles.split('>>')
        reverse_rxn_smiles = f'{p_smiles}>>{r_smiles}'
        rxn_infos.append([rxn_id+'_reverse', reverse_rxn_smiles, dH_reverse, dE_dag_reverse])

    return rxn_infos


if '__main__' == __name__:
    
    import sys
    # load in data
    try:
        csv_file = sys.argv[1]
    except:
        print ('File directory is not provided !!!')
        print ('Ex: B3LYPD3_TZVP_reaction_infos.csv')

    dH = []
    dE_dag = []
    edge_infos = read_rxn_data(csv_file)
    rxn_infos = get_rxn_infos(edge_infos)
    for rxn_info in rxn_infos:
        rxn_name, rxn_smiles, dH_val, dE_dag_val = rxn_info
        dH.append(dH_val)
        dE_dag.append(dE_dag_val)

    dH = np.array(dH)
    dE_dag = np.array(dE_dag)

    # ---- 1) Compute a 2D histogram ----
    bins = 500 
    H, xedges, yedges = np.histogram2d(dH, dE_dag, bins=bins)

    # Clip extreme counts so the color scale isn’t dominated by outliers
    vmax = np.percentile(H[H > 0], 99.7)
    vmin = 1 # minimum for log scale

    # ---- 2) Draw with a JointGrid ----
    sns.set_theme(context="talk", style="white")
    g = sns.JointGrid()

    # central 2D hist as a colored grid
    g.ax_joint.pcolormesh(
        xedges, yedges, H.T,  
        cmap="plasma",        
        norm=LogNorm(vmin=vmin, vmax=vmax),  # log counts for dense clouds
        shading="auto"
    )

    # marginal histograms
    sns.histplot(x=dH, bins=60, element="bars", linewidth=1, edgecolor="#243b6b", color="#243b6b",
                 alpha=0.25, ax=g.ax_marg_x, kde=True)
    sns.histplot(y=dE_dag, bins=60, element="bars", linewidth=1, edgecolor="#243b6b", color="#243b6b",
                 alpha=0.25, ax=g.ax_marg_y, orientation="horizontal", kde=True)

    # ---- 3) Cosmetics  ----
    for spine in g.ax_joint.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color("black")

    # labels, ticks and layout
    g.ax_joint.set_xlabel("ΔH$_r$ (kcal/mol)")
    g.ax_joint.set_ylabel("ΔE$^{\\ddagger}$ (kcal/mol)")

    g.ax_joint.tick_params(length=0)
    g.ax_joint.set_xlim(-250, 250)
    g.ax_joint.set_ylim(0, 250)
    g.ax_marg_x.set_yticks([])
    g.ax_marg_y.set_xticks([])

    plt.tight_layout()
    plt.savefig(f'dE_dag vs. dH.pdf', format='pdf')
    plt.show()


