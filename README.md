# Reaction-QM
Complementary codes for 'A Comprehensive Dataset of Chemical Reactions Covering Second and Third Row Elements with Million-Scale Quantum Calculations.'

- **draw_distribution.py**: This script plots the reaction feature distribution shown in Fig. 3. It requires the `reaction_statistics_gfn.pkl`, `reaction_statistics_dft.pkl`, and `reaction_statistics_rgd1.py` data files, which are included in the repository.
- **draw_dH_vs_dE_dag.py**: This script generates the distribution of reaction enthalpies (ΔH) and activation energies (ΔE‡) shown in Fig. 4.
- **open_h5_example.py**: This script prints the data for a single reaction from the B3LYP_TZVP.h5 file to the terminal. The script can be modified to read other data files, and a specific reaction can be targeted via a command-line argument. If no reaction is specified, the data for `RXN_0000000001` is displayed by default.

```bash
$ open_h5_example.py RXN_XXXXXXXXXX
```

- **open_h5_irc_example.py**: This script prints the data for a single IRC trajectory from the `IRC_4_15.h5` file to the terminal and plots its corresponding energy profile. Similarly, the target file can be changed within the script, and a specific transition state (TS) can be targeted via a command-line argument. By default, the IRC trajectory for `TS_0000000001` is processed.

```bash
$ open_h5_irc_example.py TS_XXXXXXXXXX
```
