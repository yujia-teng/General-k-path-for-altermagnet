# General-k-path-for-altermagnet
This is the script used for generating general k-path for band structure calculation. e.g. from `Γ−M−K−Γ` to `Γ−M−𝑘|𝑘'−M'−K'−𝑘'|𝑘−K-Γ`

Currently it fully supports VASP only. For QE user, a few more extra pre-/post-processing steps are needed. Will fully support QE soon. No plan for other codes at the moment.

## Usage
Step 1: `python3 find_operations.py`, based on `spinspg` https://github.com/spglib/spinspg, producing spin-flip operations used for next step.

Step 2: `python3 auto-generate-general-kpath.py`, generating the general k-path file `KPOINTS_modified`.

Step 3: Run the first-principle codes to perform band structure calculation based on the general k-path.

## Example
```
$ python3 find_operations.py
========================================
1. Structure Loading
========================================
Enter structure file name (default: POSCAR): 
Successfully loaded 'POSCAR' containing 6 atoms.

========================================
2. Non-Magnetic Space Group Analysis
========================================
Space Group: P6_3mc (186)

========================================
3. Magnetic Configuration Input
========================================
Enter magnetic moments (space-separated, e.g., '1 -1'):
Moments: 1 -1
Using magnetic moments:
[[ 0.  0.  1.]
 [ 0.  0. -1.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]

========================================
4. Spin Space Group Analysis
========================================
Spin-Only Group Type: COLLINEAR(axis=[0. 0. 1.])
Magnetic Space Group: Not found (spglib too old?)
Total Symmetry Operations: 12

========================================
5. Saving Results
========================================
[INFO] All operations written to 'spin_operations.txt'
[INFO] 6 spin-flipping matrices written to 'flip_spin_operations.txt'
```
Note: In step 3, we just need to give the direction of magnetic moment of magnetic atom. 0 can be skipped. Here, we take default that spin is along z axis. This doesn't matter because we are interested in operations without SOC, so spin are just black/white or positive/negative value, which is described by spin group.
```
$ python3 auto-generate-general-kpath.py 
=== Altermagnetic K-Path Generator ===
Recommend to use continues high symmetry kpath as input like G-M-K-G rather than L-M|H-K, otherwise there will be duplicated paths.

Step 1: Reading KPOINTS file...
Enter KPOINTS file name (default: KPATH.in): 
Successfully read 14 k-points from KPATH.in

Step 2: Enter general k-point coordinates
Format: kx ky kz (space-separated)
Enter k-point: 0.2777777778   0.1111111111   0.25

Step 3: Selecting Transformation Matrix R
Found 6 pre-calculated spin-flip operations:

  Option 1:
    [  1 -1  0 ]
    [  1  0  0 ]
    [  0  0  1 ]

  Option 2:
    [ -1  0  0 ]
    [  0 -1  0 ]
    [  0  0  1 ]

  Option 3:
    [  0  1  0 ]
    [ -1  1  0 ]
    [  0  0  1 ]

  Option 4:
    [  0  1  0 ]
    [  1  0  0 ]
    [  0  0  1 ]

  Option 5:
    [  1 -1  0 ]
    [  0 -1  0 ]
    [  0  0  1 ]

  Option 6:
    [ -1  0  0 ]
    [ -1  1  0 ]
    [  0  0  1 ]

Select an operation number (1-6)
Press [Enter] for default (1), or type number: 5
Selected: Option 5

Processing k-points...
Using Transformation Matrix R:
  [ 1. -1.  0.]
  [ 0. -1.  0.]
  [0. 0. 1.]
k' (transformed k): [0.2778, -0.3889, 0.2500]
Found 8 unique high-symmetry points
  0: GAMMA = (0.0000, 0.0000, 0.0000)
  1: M = (0.5000, 0.0000, 0.0000)
  2: K = (0.3333, 0.3333, 0.0000)
  3: GAMMA = (0.0000, 0.0000, 0.0000)
  4: A = (0.0000, 0.0000, 0.5000)
  5: L = (0.5000, 0.0000, 0.5000)
  6: H = (0.3333, 0.3333, 0.5000)
  7: A = (0.0000, 0.0000, 0.5000)

Modified k-points (original: 14, new: 28):

Enter output filename (default: KPOINTS_modified): 
Modified KPOINTS file written to: KPOINTS_modified

Process completed successfully!
```
Note: 1. Currently, we need to manually type the k-point coordinate (`0.277777778 0.1111111111 0.25` here), which should be the average/centroid of high-sym points in irr BZ. We are still working on this to make the whole procedure more automated. 
2. In output `k_t` is just `k'`. I use `k_t` for now instead of `k'` is because of format problem. Will solve this later.
