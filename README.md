# AlterSeeK-Path

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22133630.svg)](https://doi.org/10.5281/zenodo.22133630)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

AlterSeeK-Path generates general-k paths for altermagnet band-structure calculations. It inserts a general k point `k` and its spin-flip partner `k'` into a standard high-symmetry path, using the IBZ centroid as the default general point.

![AlterSeeK-Path example](./example/GdAuGe/VASP/HEX.png)

---

## Installation

Requires Python >= 3.11.

```bash
pip install alterseek-path
```

To install the latest development version from source instead:

```bash
git clone https://github.com/yujia-teng/AlterSeeK-Path.git
cd AlterSeeK-Path
pip install .
```

---

## Documentation

For the full user guide, see
[yujia-teng.github.io/AlterSeeK-Path](https://yujia-teng.github.io/AlterSeeK-Path/).

The case library - VASP inputs, generated k-paths, and band-structure data for
54 three-dimensional and 12 two-dimensional cases - is archived at [doi.org/10.5281/zenodo.22133631](https://doi.org/10.5281/zenodo.22133631).

---

## Quick Start

```bash
alterseek-path
```

This runs interactively, prompting for the structure file, spin axis,
moments, k-path, and so on. The default output file is:

```text
KPOINTS_alter
```

To skip the prompts (e.g. for repeated runs on the same structure), put an
`alterseek_input.toml` file in the working directory:

```toml
structure = "POSCAR"
spin_axis = "0 0 1"
moments = "5 -5"
symprec = 1e-3
flip_option = 1
output_code = "vasp"
save_pdf = false
view_elev = 14
view_azim = 20
# vacuum_axis = "c"  # 2D only
```

`alterseek-path` reads any keys it finds and only prompts for the ones that
are missing. See [Workflow](https://yujia-teng.github.io/AlterSeeK-Path/workflow/#input-configuration)
for the full field reference.

---

## Inputs

- Structure file: `POSCAR` / `.vasp` / `.cif` (moments entered manually) or
  `.mcif` (moments read from the file). MCIF input must be collinear: all
  nonzero vector moments must be parallel or antiparallel within an absolute
  transverse-moment tolerance of `0.02` in the MCIF moment units (normally
  Bohr magnetons); noncollinear MCIFs are rejected.
- Spin axis + moments: Cartesian axis (default `0 0 1`) and VASP
  `MAGMOM`-style scalar moments (e.g. `5*0 2*1.0`); missing trailing values
  default to `0`, while excess values are rejected.

These (plus the Step 3 operation choice and Step 5 output code) can also be
supplied via `alterseek_input.toml` — see Quick Start above.

---

## Example Run

```text
=== AlterSeeK-Path 1.0.1 ===

>>> Step 0: Spin symmetry
Enter structure file (default: POSCAR, supports .vasp/.cif/.mcif): POSCAR
Spin axis in Cartesian coordinates (default: 0 0 1): 0 0 1
Magnetic moments along this axis (atom order, trailing atoms auto-fill to 0): 8 -8 4*0

Input structure: POSCAR, 6 atoms
Input cell:                   SG P6_3mc (186)  PG 6mm  Laue 6/mmm  [6 atoms, hP2]
Nonmagnetic primitive cell:   SG P6_3mc (186)  PG 6mm  Laue 6/mmm  [6 atoms, hP2]
Magnetic primitive cell:      SG P6_3mc (186)  PG 6mm  Laue 6/mmm  [6 atoms, hP2]
Phase: AFM(Altermagnet)
Oriented SSG: 186.156.1.1.L
SSG Symbol (Chen-Liu): P -1|6_{3} 1|m -1|c infinity_{001}m|1
MSG without SOC: P6_3'mc' (BNS 186.206), Type III
Spin operations: 6 flip, 6 preserve

>>> Step 1: High-symmetry k-path
Path: GAMMA-M-K-GAMMA-A-L-H-A | L-M | H-K
Using HPKOT hP2 path (9 segments, 18 k-points)

>>> Step 2: General k-point
IBZ centroid (standardized basis): [0.277778, 0.111111, 0.250000]
IBZ centroid (input-cell basis): [0.277778, 0.111111, 0.250000]

>>> Step 3: Spin-flip operation
Found 12 spin-flip operations R.
  Note: R is in the submitted structure 'POSCAR' fractional basis;
  rotation axis/mirror plane indices are in the reciprocal (b1,b2,b3) basis.
Default R: Option 1
Press [Enter] to use default, type a number, or 'list' to show matrices: 1
Selected: Option 1  (C6+ [0 0 1])

>>> Step 4: Build general-k path
k' = [-0.1111, 0.3889, 0.2500]
Generated path: GAMMA-M-k | k'-M'-K'-k' | k-K-GAMMA-k | ... | k-H-A | L-M | H-K
Full path: 9 original segments -> 21 generated segments, 36 k-points

>>> Step 5: Save
Output code ([vasp]/qe/abinit): vasp
Modified KPOINTS file written to: KPOINTS_alter
Band plot config updated: alterseek_plot_vasp.toml (lattice_type = "hP2")

Done.
Displaying generated figure(s)...
Saved: alterseek_output\POSCAR_ibz_hP2.png  (view_elev = 14.00, view_azim = 20.00)
Saved: alterseek_output\POSCAR_spinflip_hP2.png  (view_elev = 14.00, view_azim = 20.00)
Saved: alterseek_output\POSCAR_spinbz_hP2.png  (view_elev = 14.00, view_azim = 20.00)
Saved: alterseek_output\POSCAR_spinbz_top_hP2.png
Saved: alterseek_output\figure_camera_angle.txt
Run log: alterseek_output\alterseek_run.log
```

---

## 2D / Slab Mode

For 2D materials computed as slabs (vacuum along one lattice vector), run:

```bash
alterseek-path --2d
```

2D mode restricts the k-path and IBZ centroid to the physical in-plane
(vacuum k = 0) reciprocal plane, and reports whether any spin-flip operation
produces in-plane spin splitting. See [Workflow](https://yujia-teng.github.io/AlterSeeK-Path/workflow/#2d-slab-mode)
for more details.

---

## Cell Setting and Brillouin Zone

AlterSeeK-Path performs the analysis in the input cell without converting it
to a primitive cell. For a conventional cell or supercell, its lattice vectors
define the folded BZ, and the IBZ, centroid, path, and figures are constructed
in that cell setting. For example, a conventional cubic fcc input uses the
simple-cubic calculation-cell BZ rather than the primitive fcc BZ.

Magnetic symmetry is also included. For example, if magnetic order lowers a
hexagonal parent structure to an orthorhombic G0, AlterSeeK-Path uses the
orthorhombic symmetry to determine the IBZ and path while retaining the
input-cell basis.

---

## Band Plotting

After the band calculation and code-specific post-processing, run:

```bash
alterseek-plot
```

The command detects `vasp`, `qe`, or `abinit` when exactly one generated
`alterseek_plot_*.toml` file is present. To select explicitly, run
`alterseek-plot vasp`, `alterseek-plot qe`, or `alterseek-plot abinit`. See
[Plotting](https://yujia-teng.github.io/AlterSeeK-Path/plotting/) for the
required band files and settings.

---

## Citation

```bibtex
@misc{teng2026alterseekpath,
  title = {AlterSeeK-Path: Systematic construction of generalized band-structure paths for displaying altermagnetic spin splitting},
  author = {Yujia Teng and Mesfin Eshete and Andrea Urru and Daniel Seleznev and Se Young Park and Sebastian E. Reyes-Lillo and Karin M. Rabe},
  year = {2026},
  archivePrefix = {arXiv},
  eprint = {2609.02770},
  primaryClass = {cond-mat.mtrl-sci},
  url = {https://arxiv.org/abs/2609.02770}
}
```

```bibtex
@article{v3fg-6smc,
  title = {$G$-type antiferromagnetic ${\mathrm{BiFeO}}_{3}$ is a multiferroic $g$-wave altermagnet},
  author = {Urru, Andrea and Seleznev, Daniel and Teng, Yujia and Park, Se Young and Reyes-Lillo, Sebastian E. and Rabe, Karin M.},
  journal = {Phys. Rev. B},
  volume = {112},
  issue = {10},
  pages = {104411},
  numpages = {14},
  year = {2025},
  month = {Sep},
  publisher = {American Physical Society},
  doi = {10.1103/v3fg-6smc},
  url = {https://link.aps.org/doi/10.1103/v3fg-6smc}
}
```
