# Reaction-QM

Complementary codes for 'A Comprehensive Dataset of Chemical Reactions Covering Second and Third Row Elements with Million-Scale Quantum Chemical Calculations.'

# Requirements

- **Python:** ≥ 3.7
- **NumPy**
- **h5py**: ≥ 2.10.0
- **HDF5 C Library**: 1.10.4
- **Matplotlib**

You may need these additional packages to reproduce the figures:

- **pandas**
- **Seaborn**

# How to Use the Dataset

To facilitate easy access to our HDF5 dataset, we have included exemplary scripts in the `h5_examples` folder. You can use these to learn how to open the files and browse the stored reaction data.

1. Download the dataset from [Zenodo](https://zenodo.org/records/18551029).
2. Run the examples in `h5_examples` to understand the data retrieval process.

For a detailed explanation of the data schema and parameters, please refer to our original paper.

### `open_h5_example.py`

This script demonstrates how to retrieve and inspect reaction data from the `B3LYPD3_TZVP.h5` and `GFN2_xTB.h5` files.

- **Functionality:** It prints comprehensive information for a single reaction entry to the terminal, including energies, charges, spin multiplicities and atomic structures (coordinates) for reactants, products, and a transition state (TS).
- **Arguments:**
    - `path`: (Required) Path to the HDF5 file.
    - `rxn_id`: (Optional) The target reaction ID (e.g., `0000000001` or `RXN_0000000001`).
- **Default Behavior:** If no reaction ID is specified, the script displays data for `RXN_0000000001` by default.
- **Usage Example:**
    
    ```bash
    python open_h5_example.py path/to/hdf5_file RXN_XXXXXXXXXX
    python open_h5_example.py path/to/hdf5_file XXXXXXXXXX
    ```
    
    ```bash
    ---- Information of RXN_0000000001 ----                             # Target reaction
    -------------------- P0 --------------------                        # P0 indicates that it is the first product
    SMILES: [Cl:1]/[C:2](=[C:3](\[Cl:4])[C:5]([Cl:6])([H:8])[H:9])[H:7] # Molecule SMILES
    E, H, G (Hartree): [-1496.81194528 -1496.750949   -1496.79139   ]   # Single point energy, enthalpy, and Gibbs free energy
    Charge : 0                                                          # Total charge of the molecule
    Spin multiplicity : 1                                               # Spin multiplicity of the molecule
    xyz coordinates (Å):                                                # The atomic number and its coordinate pairs, which looks like XYZ format
    17 2.173693895339966 -1.2708020210266113 -0.00038499999209307134
    6 0.48249301314353943 -0.8669030070304871 -3.999999989900971e-06
    6 -0.012853999622166157 0.36268100142478943 -3.999999989900971e-06
    17 1.0006860494613647 1.799090027809143 -0.00015799999528098851
    6 -1.4663029909133911 0.7405700087547302 0.00035200000274926424
    17 -2.6129350662231445 -0.6668949723243713 0.0003650000144261867
    1 -0.17301200330257416 -1.7239710092544556 -3.999999989900971e-06
    1 -1.6959480047225952 1.331231951713562 -0.8849530220031738
    1 -1.6956050395965576 1.3309619426727295 0.8859279751777649
    -------------------- P1 --------------------
    SMILES: [O:1]=[C:2]=[S:3]
    E, H, G (Hartree): [-511.60313098 -511.590172   -511.616464  ]
    Charge : 0
    Spin multiplicity : 1
    xyz coordinates (Å):
    8 0.0 0.0 -1.6863789558410645
    6 0.0 0.0 -0.5296980142593384
    16 0.0 0.0 1.0418260097503662
    -------------------- R0 --------------------
    SMILES: [Cl:1][C@@:2](/[C:3]([Cl:4])=[C:5](\[Cl:6])[H:12])([C:8](=[O:7])[S:9][H:11])[H:10]
    E, H, G (Hartree): [-2008.39497467 -2008.321712   -2008.371614  ]
    Charge : 0
    Spin multiplicity : 1
    xyz coordinates (Å):
    17 0.6631320118904114 -0.8048539757728577 1.8243780136108398
    6 0.08068700134754181 -0.5453659892082214 0.11371699720621109
    6 -0.8699970245361328 0.6030340194702148 0.04790399968624115
    17 -0.16688600182533264 2.201493978500366 0.2681800127029419
    6 -2.18013596534729 0.521681010723114 -0.15616199374198914
    17 -3.0438969135284424 -0.9780060052871704 -0.37790900468826294
    8 0.9145219922065735 -0.3107509911060333 -2.0868399143218994
    6 1.2119460105895996 -0.38702499866485596 -0.9267870187759399
    16 2.90014910697937 -0.3454340100288391 -0.31300899386405945
    1 -0.4446449875831604 -1.4579579830169678 -0.15854200720787048
    1 3.3947060108184814 -0.19454999268054962 -1.5622650384902954
    1 -2.8135499954223633 1.3947529792785645 -0.19739599525928497
    -------------------- TS --------------------                     # For TS, full reaction SMILES is saved in the 'smiles' dataset
    SMILES: [Cl:1][C@@:2](/[C:3]([Cl:4])=[C:5](\[Cl:6])[H:12])([C:8](=[O:7])[S:9][H:11])[H:10]>>[Cl:1]/[C:2](=[C:3](\[Cl:4])[C:5]([Cl:6])([H:11])[H:12])[H:10].[O:7]=[C:8]=[S:9]
    E, H, G (Hartree): [-2008.33796771 -2008.269416   -2008.317115  ]
    Charge : 0
    Spin multiplicity : 1
    xyz coordinates (Å):
    17 2.291088104248047 -0.438616007566452 -1.082921028137207
    6 0.5992590188980103 -0.12639999389648438 -0.7246729731559753
    6 -0.10445199906826019 -0.9748889803886414 0.1585109978914261
    17 0.734486997127533 -1.7657129764556885 1.4580190181732178
    6 -1.4881420135498047 -0.8514869809150696 0.3098889887332916
    17 -2.5243170261383057 -0.600691020488739 -1.119994044303894
    8 1.5238929986953735 2.2514870166778564 -0.17545799911022186
    6 0.5342220067977905 1.673097014427185 0.10200200229883194
    16 -0.9149420261383057 2.013964891433716 0.9231600165367126
    1 0.056154001504182816 0.1305759996175766 -1.6267789602279663
    1 -1.3897379636764526 0.44961801171302795 0.7889760136604309
    1 -1.9851959943771362 -1.4521069526672363 1.059762954711914
    
    ```
    

### `open_h5_irc_example.py`

This script is designed for analyzing **Intrinsic Reaction Coordinate (IRC) trajectories** from the B3LYP-RXN dataset, specifically stored in `B3LYPD3_TZVP_IRC.h5` .

- **Functionality:** It opens a single IRC path, prints all associated metadata, and generates an energy profile visualization using **Matplotlib** .
- **Arguments:**
    - `path`: (Required) Path to the IRC HDF5 file.
    - `rxn_id`: (Optional) The target reaction ID (e.g., `0000000001` or `RXN_0000000001`).
- **Path Reordering:** Since the dataset simply concatenates backward and forward IRC results, this script demonstrates how to rearrange the steps to ensure a continuous and physical reaction coordinate.
- **Energy Profile**: You can view the energy profile plot by deleting the '#' in the last line of the code. The interactive plot window will not appear if your remote server environment is not compatible.
- **Default Behavior:** If no reaction ID is specified, the script displays data for `RXN_0000000001` by default.
- **Usage Example:**
    
    ```bash
    python open_h5_irc_example.py path/to/irc_hdf5_file RXN_XXXXXXXXXX
    python open_h5_irc_example.py path/to/irc_hdf5_file XXXXXXXXXX
    ```
    
    ```bash
    ---- RXN_0000000001 ----
    Atomic numbers: [17  6  6 17  6 17  8  6 16  1  1  1] # This order applies to both coordinates and forces
    Coordinates: [[[ 2.291088 -0.438615 -1.082921]
      [ 0.599259 -0.1264   -0.724673]
      [-0.104452 -0.974889  0.158511]
      ...
      [ 0.056154  0.130576 -1.626779]
      [-1.389738  0.449618  0.788976]
      [-1.985196 -1.452108  1.059763]]
    
     [[ 2.272503 -0.429108 -1.078113]
      [ 0.580517 -0.123136 -0.721123]
      [-0.118234 -0.965927  0.16121 ]
      ...
      [ 0.036909  0.144538 -1.61971 ]
      [-1.427185  0.385346  0.781069]
      [-2.001183 -1.450509  1.060065]]
    
     [[ 2.272546 -0.429013 -1.078045]
      [ 0.580412 -0.130014 -0.722544]
      [-0.112992 -0.96636   0.158992]
      ...
      [ 0.036243  0.149807 -1.617004]
      [-1.450018  0.315059  0.772441]
      [-1.998314 -1.46023   1.05464 ]]
    
     ...
    
     [[ 2.062767 -0.432168 -1.226116]
      [ 0.475277  0.153236 -0.581131]
      [-0.204151 -0.911912  0.222383]
      ...
      [-0.141671  0.364677 -1.45421 ]
      [-1.625768  1.149879  0.807753]
      [-1.881178 -2.14205   0.624758]]
    
     [[ 2.058028 -0.431754 -1.231903]
      [ 0.473487  0.155151 -0.581353]
      [-0.204942 -0.910344  0.222411]
      ...
      [-0.145705  0.370014 -1.45197 ]
      [-1.630366  1.157858  0.794615]
      [-1.879542 -2.143032  0.625142]]
    
     [[ 2.053423 -0.431534 -1.237988]
      [ 0.471859  0.157245 -0.581053]
      [-0.205612 -0.908599  0.222957]
      ...
      [-0.149681  0.374272 -1.449454]
      [-1.632267  1.166183  0.785352]
      [-1.878586 -2.144402  0.62525 ]]]
    Energies: [-54649.65774989 -54649.68387418 -54649.74875891 -54649.82829208
     -54649.91446101 -54650.00534486 -54650.09917216 -54650.19469174
     -54650.29069106 -54650.38616602 -54650.4801555  -54650.57153949
     -54650.66002167 -54650.7454461  -54650.82865824 -54650.90813752
     -54650.98364395 -54651.0548839  -54651.12171914 -54651.18404029
     -54651.24171319 -54651.29478301 -54651.34254497 -54651.38709709
     -54651.42707605 -54651.46443347 -54651.49742156 -54651.52765205
     -54651.55468439 -54651.5801137  -54651.6043819  -54651.626477
     -54651.64782433 -54651.66762361 -54651.68623864 -54651.703581
     -54651.71973096 -54651.73475573 -54651.74868632 -54651.76159758
     -54651.77350746 -54651.78447256 -54651.79447193 -54651.8034789
     -54651.81139496 -54651.81853931 -54651.82519984 -54651.83130363
     -54651.83698319 -54651.84172423 -54651.84659588 -54651.85078371
     -54651.85460664 -54651.85789486 -54651.8606609  -54651.86274584
     -54651.86471567 -54651.86691898 -54651.86883357 -54651.8705038
     -54651.87176015 -54649.68465324 -54649.75244578 -54649.83536976
     -54649.92085895 -54650.00755197 -54650.09367955 -54650.17794613
     -54650.25930899 -54650.33689627 -54650.40994523 -54650.47780552
     -54650.53995851 -54650.59608552 -54650.64613542 -54650.6903659
     -54650.72918865 -54650.76289349 -54650.79230927 -54650.82010407
     -54650.84592522 -54650.86923912 -54650.89032033 -54650.90932258
     -54650.92634004 -54650.94129024 -54650.95405674 -54650.96412713
     -54650.97369574 -54650.98225426 -54650.98983617 -54650.99632908
     -54650.99916614 -54651.00537115 -54651.0100009  -54651.01446601
     -54651.01885494 -54651.02306154 -54651.02716257 -54651.03114741
     -54651.03506149 -54651.03888333 -54651.04265129 -54651.04634279
     -54651.04998748 -54651.05357231 -54651.05710952 -54651.0605953
     -54651.06403318 -54651.06742127 -54651.07075494 -54651.07402003
     -54651.07721601 -54651.08031974 -54651.08343871 -54651.08651985
     -54651.08958984 -54651.09260405 -54651.09557118 -54651.09847028
     -54651.10131278]
    Forces: [[[ 5.7181340e-05 -7.1579518e-05 -2.6482365e-05]
      [-1.2367008e-04 -7.5744705e-05 -1.3287461e-04]
      [ 2.6677767e-04 -1.8270260e-04  1.2145892e-04]
      ...
      [-1.5478043e-05 -1.1246006e-04 -1.4706711e-05]
      [-3.0375016e-04 -4.0880544e-04  3.8566552e-05]
      [-4.2166095e-05  2.1288735e-05 -2.3602728e-05]]
    
     [[ 7.7936314e-03  1.8085398e-02  1.3510840e-02]
      [-8.8464981e-03 -4.5622650e-01 -9.2684731e-02]
      [ 3.6444315e-01 -3.0045811e-02 -1.5183157e-01]
      ...
      [-2.9685446e-03  2.9785357e-02  1.6585622e-02]
      [-1.3616949e-01 -3.8337934e-01 -4.0145412e-02]
      [ 1.8264655e-02 -5.0535038e-02 -3.3092517e-02]]
    
     [[ 1.9353054e-02  3.3665873e-02  2.3010038e-02]
      [-7.6810168e-03 -8.8109148e-01 -2.0082667e-01]
      [ 6.5254575e-01 -4.2258244e-02 -2.7274284e-01]
      ...
      [-5.0309296e-03  5.8140103e-02  3.3652298e-02]
      [-2.5355837e-01 -4.7777420e-01  1.5844271e-02]
      [ 3.5425384e-02 -1.0994701e-01 -7.1568720e-02]]
    
     ...
    
     [[-5.5746663e-02  3.2340367e-03 -6.8443798e-02]
      [-5.2692192e-03  1.1388188e-02 -5.7445136e-03]
      [-5.6562219e-03  4.5414940e-03  2.3784249e-03]
      ...
      [-1.2313013e-03  2.5453922e-04  4.2042683e-04]
      [-1.0664268e-02 -1.1955990e-02 -4.8306403e-03]
      [ 1.3104400e-03 -1.6105906e-03  3.6355403e-05]]
    
     [[-5.3768817e-02  3.0796162e-03 -6.9060348e-02]
      [-6.2317373e-03  9.0377880e-03  8.8832648e-03]
      [-8.9582382e-04  7.3275929e-03  1.3467440e-04]
      ...
      [-1.2464709e-04  1.8512459e-03  2.0716409e-03]
      [ 1.4526065e-02  1.7857548e-02  2.8978905e-03]
      [-1.0965242e-03 -1.4842466e-03 -9.8689226e-04]]
    
     [[-5.5786774e-02  5.3519574e-03 -6.4424857e-02]
      [-8.4728654e-03  3.8353149e-03 -1.3917640e-02]
      [-5.8578989e-03  5.2752355e-03  3.1830259e-03]
      ...
      [-1.1086598e-04  1.7072127e-05  8.5586886e-04]
      [-1.4142148e-02 -1.5199540e-02 -5.5213929e-03]
      [ 4.2401096e-03  2.3548736e-03 -2.8585526e-03]]]
    ```
    <img width="800" height="800" alt="irc_open_example" src="https://github.com/user-attachments/assets/20e31ec6-afd3-4ca6-8610-3be67868175f" />



# Drawing Figures

We provide scripts to reproduce the figures presented in our paper.

- **Figure 2: Feature Distribution**
    - **`draw_distribution.py`** : Plots the reaction feature distributions shown in **Fig. 2.** We uploaded the sample statistics result, `reaction_statistics_dft.pkl`, to test the plotting code.
- **Figure 4: Energetics & Correlation**
    - **`draw_dH_vs_dE_dag.py`** : Generates the distribution of reaction enthalpies and activation energies. Requires the energetics CSV file from **Zenodo.**
    - **`plot_correlation.py`** : Compares reaction enthalpies and activation energies calculated at the **B3LYP-D3/TZVP** and **GFN2-xTB** levels of theory. Uses `common_reaction_info.csv` from **Zenodo**.
    - **`plot_rmsd.py`** : Visualizes the **Root Mean Square Distance (RMSD)** between transition state geometries from different levels of theory using `ts_rmsd.csv` (included in this repository).
- **Figure 5: Forces & Atomic Distances**
    - `draw_force_histogram.py` and `draw_interatomic_distance_heatmap.py`: Plots the force and distance distribution shown in **Fig. 5**. Due to the large size of the original dataset, we provide a small sample of distances and forces in the same folder. Users must run **`get_r_zeros.py`** before drawing the distributions to process the necessary reference data.
