Usage
=====

To dock ligand `lig.sdf` to a binding site on `rec.pdb` defined by another ligand `orig.sdf`:
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf -o docked.sdf.gz
```

To perform docking with flexible sidechain residues within 3.5 Angstroms of `orig.sdf` (generally not recommend unless prior knowledge indicates pocket is highly flexible):
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --flexdist_ligand orig.sdf --flexdist 3.5 -o flex_docked.sdf.gz
```

To perform whole protein docking:
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand rec.pdb -o whole_docked.sdf.gz --exhaustiveness 64
```

To utilize the default ensemble CNN in the energy minimization during the refinement step of docking (10 times slower than the default rescore option):
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --cnn_scoring refinement -o cnn_refined.sdf.gz
```

To utilize the default ensemble CNN for every step of docking (1000 times slower than the default rescore option):
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --cnn_scoring all -o cnn_all.sdf.gz
```

To utilize all empirical scoring using the [Vinardo](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155183) scoring function:
```
gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --scoring vinardo --cnn_scoring none -o vinardo_docked.sdf.gz
```

To utilize a different CNN during docking (see help for possible options):
```

gnina -r rec.pdb -l lig.sdf --autobox_ligand orig.sdf --cnn dense -o dense_docked.sdf.gz
```

To minimize and score ligands `ligs.sdf` already positioned in a binding site:
```
gnina -r rec.pdb -l ligs.sdf --minimize -o minimized.sdf.gz
```

To covalently dock a pyrazole to a specific iron atom on the receptor with the bond formed between a nitrogen of the pyrazole and the iron.
```
gnina  -r rec.pdb.gz -l conformer.sdf.gz --autobox_ligand bindingsite.sdf.gz --covalent_rec_atom A:601:FE --covalent_lig_atom_pattern '[$(n1nccc1)]' -o output.sdf.gz 
```

The same as above, but with the covalently bonding ligand atom manually positioned (instead of using OpenBabel binding heuristics) and the ligand/residue complex UFF optimized.
```
gnina  -r rec.pdb.gz -l conformer.sdf.gz --autobox_ligand bindingsite.sdf.gz --covalent_lig_atom_position -11.796,31.887,72.682  --covalent_optimize_lig  --covalent_rec_atom A:601:FE --covalent_lig_atom_pattern '[$(n1nccc1)]' -o output.sdf.gz 
```

All options:
```
Input:
  -r [ --receptor ] arg              rigid part of the receptor
  --flex arg                         flexible side chains, if any (PDBQT)
  -l [ --ligand ] arg                ligand(s)
  --flexres arg                      flexible side chains specified by comma 
                                     separated list of chain:resid
  --flexdist_ligand arg              Ligand to use for flexdist
  --flexdist arg                     set all side chains within specified 
                                     distance to flexdist_ligand to flexible
  --flex_limit arg                   Hard limit for the number of flexible 
                                     residues
  --flex_max arg                     Retain at at most the closest flex_max 
                                     flexible residues

Search space (required):
  --center_x arg                     X coordinate of the center
  --center_y arg                     Y coordinate of the center
  --center_z arg                     Z coordinate of the center
  --size_x arg                       size in the X dimension (Angstroms)
  --size_y arg                       size in the Y dimension (Angstroms)
  --size_z arg                       size in the Z dimension (Angstroms)
  --autobox_ligand arg               Ligand to use for autobox. A multi-ligand 
                                     file still only defines a single box.
  --autobox_add arg                  Amount of buffer space to add to 
                                     auto-generated box (default +4 on all six 
                                     sides)
  --autobox_extend arg (=1)          Expand the autobox if needed to ensure the
                                     input conformation of the ligand being 
                                     docked can freely rotate within the box.
  --no_lig                           no ligand; for sampling/minimizing 
                                     flexible residues

Covalent docking:
  --covalent_rec_atom arg            Receptor atom ligand is covalently bound 
                                     to.  Can be specified as 
                                     chain:resnum:atom_name or as x,y,z 
                                     Cartesian coordinates.
  --covalent_lig_atom_pattern arg    SMARTS expression for ligand atom that 
                                     will covalently bind protein.
  --covalent_lig_atom_position arg   Optional.  Initial placement of covalently
                                     bonding ligand atom in x,y,z Cartesian 
                                     coordinates.  If not specified, 
                                     OpenBabel's GetNewBondVector function will
                                     be used to position ligand.
  --covalent_fix_lig_atom_position   If covalent_lig_atom_position is 
                                     specified, fix the ligand atom to this 
                                     position as opposed to using this position
                                     to define the initial structure.
  --covalent_bond_order arg (=1)     Bond order of covalent bond. Default 1.
  --covalent_optimize_lig            Optimize the covalent complex of ligand 
                                     and residue using UFF. This will change 
                                     bond angles and lengths of the ligand.

Scoring and minimization options:
  --scoring arg                      specify alternative built-in scoring 
                                     function: ad4_scoring default dkoes_fast 
                                     dkoes_scoring dkoes_scoring_old vina 
                                     vinardo
  --custom_scoring arg               custom scoring function file
  --custom_atoms arg                 custom atom type parameters file
  --score_only                       score provided ligand pose
  --local_only                       local search only using autobox (you 
                                     probably want to use --minimize)
  --minimize                         energy minimization
  --randomize_only                   generate random poses, attempting to avoid
                                     clashes
  --num_mc_steps arg                 fixed number of monte carlo steps to take 
                                     in each chain
  --max_mc_steps arg                 cap on number of monte carlo steps to take
                                     in each chain
  --num_mc_saved arg                 number of top poses saved in each monte 
                                     carlo chain
  --temperature arg                  temperature for metropolis accept 
                                     criterion
  --minimize_iters arg (=0)          number iterations of steepest descent; 
                                     default scales with rotors and usually 
                                     isn't sufficient for convergence
  --accurate_line                    use accurate line search
  --simple_ascent                    use simple gradient ascent
  --minimize_early_term              Stop minimization before convergence 
                                     conditions are fully met.
  --minimize_single_full             During docking perform a single full 
                                     minimization instead of a truncated 
                                     pre-evaluate followed by a full.
  --approximation arg                approximation (linear, spline, or exact) 
                                     to use
  --factor arg                       approximation factor: higher results in a 
                                     finer-grained approximation
  --force_cap arg                    max allowed force; lower values more 
                                     gently minimize clashing structures
  --user_grid arg                    Autodock map file for user grid data based
                                     calculations
  --user_grid_lambda arg (=-1)       Scales user_grid and functional scoring
  --print_terms                      Print all available terms with default 
                                     parameterizations
  --print_atom_types                 Print all available atom types

Convolutional neural net (CNN) scoring:
  --cnn_scoring arg (=1)             Amount of CNN scoring: none, rescore 
                                     (default), refinement, metrorescore 
                                     (metropolis+rescore), metrorefine 
                                     (metropolis+refine), all
  --cnn arg                          built-in model to use, specify 
                                     PREFIX_ensemble to evaluate an ensemble of
                                     models starting with PREFIX: 
                                     all_default_to_default_1_3_1 
                                     all_default_to_default_1_3_2 
                                     all_default_to_default_1_3_3 
                                     crossdock_default2018 
                                     crossdock_default2018_1 
                                     crossdock_default2018_1_3 
                                     crossdock_default2018_1_3_1 
                                     crossdock_default2018_1_3_2 
                                     crossdock_default2018_1_3_3 
                                     crossdock_default2018_1_3_4 
                                     crossdock_default2018_2 
                                     crossdock_default2018_3 
                                     crossdock_default2018_4 
                                     crossdock_default2018_KD_1 
                                     crossdock_default2018_KD_2 
                                     crossdock_default2018_KD_3 
                                     crossdock_default2018_KD_4 
                                     crossdock_default2018_KD_5 default1.0 
                                     default2017 dense dense_1 dense_1_3 
                                     dense_1_3_1 dense_1_3_2 dense_1_3_3 
                                     dense_1_3_4 dense_1_3_PT_KD 
                                     dense_1_3_PT_KD_1 dense_1_3_PT_KD_2 
                                     dense_1_3_PT_KD_3 dense_1_3_PT_KD_4 
                                     dense_1_3_PT_KD_def2018 
                                     dense_1_3_PT_KD_def2018_1 
                                     dense_1_3_PT_KD_def2018_2 
                                     dense_1_3_PT_KD_def2018_3 
                                     dense_1_3_PT_KD_def2018_4 dense_2 dense_3 
                                     dense_4 fast general_default2018 
                                     general_default2018_1 
                                     general_default2018_2 
                                     general_default2018_3 
                                     general_default2018_4 
                                     general_default2018_KD_1 
                                     general_default2018_KD_2 
                                     general_default2018_KD_3 
                                     general_default2018_KD_4 
                                     general_default2018_KD_5 
                                     redock_default2018 redock_default2018_1 
                                     redock_default2018_1_3 
                                     redock_default2018_1_3_1 
                                     redock_default2018_1_3_2 
                                     redock_default2018_1_3_3 
                                     redock_default2018_1_3_4 
                                     redock_default2018_2 redock_default2018_3 
                                     redock_default2018_4 redock_default2018_KD
                                     _1 redock_default2018_KD_2 
                                     redock_default2018_KD_3 
                                     redock_default2018_KD_4 
                                     redock_default2018_KD_5
  --cnn_model arg                    torch cnn model file; if not specified a 
                                     default model ensemble will be used
  --cnn_rotation arg (=0)            evaluate multiple rotations of pose (max 
                                     24)
  --cnn_mix_emp_force                Merge CNN and empirical minus forces
  --cnn_mix_emp_energy               Merge CNN and empirical energy
  --cnn_empirical_weight arg (=1)    Weight for scaling and merging empirical 
                                     force and energy 
  --cnn_center_x arg                 X coordinate of the CNN center
  --cnn_center_y arg                 Y coordinate of the CNN center
  --cnn_center_z arg                 Z coordinate of the CNN center
  --cnn_verbose                      Enable verbose output for CNN debugging

Output:
  -o [ --out ] arg                   output file name, format taken from file 
                                     extension
  --out_flex arg                     output file for flexible receptor residues
  --log arg                          optionally, write log file
  --atom_terms arg                   optionally write per-atom interaction term
                                     values
  --atom_term_data                   embedded per-atom interaction terms in 
                                     output sd data
  --pose_sort_order arg (=0)         How to sort docking results: CNNscore 
                                     (default), CNNaffinity, Energy
  --full_flex_output                 Output entire structure for out_flex, not 
                                     just flexible residues.

Misc (optional):
  --cpu arg                          the number of CPUs to use (the default is 
                                     to try to detect the number of CPUs or, 
                                     failing that, use 1)
  --seed arg                         explicit random seed
  --exhaustiveness arg (=8)          exhaustiveness of the global search 
                                     (roughly proportional to time)
  --num_modes arg (=9)               maximum number of binding modes to 
                                     generate
  --min_rmsd_filter arg (=1)         rmsd value used to filter final poses to 
                                     remove redundancy
  -q [ --quiet ]                     Suppress output messages
  --addH arg                         automatically add hydrogens in ligands (on
                                     by default)
  --stripH arg                       remove polar hydrogens from molecule 
                                     _after_ performing atom typing for 
                                     efficiency (off by default - nonpolar are 
                                     always removed)
  --device arg (=0)                  GPU device to use
  --no_gpu                           Disable GPU acceleration, even if 
                                     available.

Configuration file (optional):
  --config arg                       the above options can be put here

Information (optional):
  --help                             display usage summary
  --help_hidden                      display usage summary with hidden options
  --version                          display program version
```