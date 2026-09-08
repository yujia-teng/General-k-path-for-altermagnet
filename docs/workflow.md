# Workflow

Run the main workflow with:

```bash
alterseek-path
```

It prompts for any input not supplied in `alterseek_input.toml`.

| Step | What it does | Input needed |
|------|--------------|--------------|
| **0** | Finds spin-flip symmetry operations and prints a compact symmetry summary | Structure file; Cartesian spin axis and magnetic moments for non-mcif inputs |
| **1** | Generates the high-symmetry path | Automatic |
| **2** | Computes the general k point | Automatic IBZ centroid |
| **3** | Selects a detected spin-flip operation | Press Enter for default, enter a number, or type `list` |
| **4** | Builds the general-k path | Automatic |
| **5** | Saves the output | Output code (`vasp`, `qe`, or `abinit`); filenames are fixed |

## Step 0: Spin Symmetry

Reads the structure and moments, then identifies the magnetic phase, spin
space group, and spin-flip/spin-preserving operations. Supported formats are
`POSCAR`, `.vasp`, `.cif`, and `.mcif`.

Only collinear magnetism is supported. For `.mcif` input, every nonzero moment
must be parallel or antiparallel to one axis within a transverse-moment
tolerance of `0.02` in the file's moment units. Noncollinear input is rejected.

For non-`.mcif` input, enter a Cartesian spin axis (default `0 0 1`) and
scalar moments along it in atom order, VASP `MAGMOM`-style (`1 -1` or
`5*0 2*1.0`). Missing trailing values default to `0`; providing more moments
than structure sites is rejected as an atom-order/count error.

## Step 1: High-Symmetry K-Path

The path is generated from the detected symmetry and reported with its
SeeK-path lattice tag (`hP2`, `oI3`, `mC2`, etc.). Output coordinates use the
submitted cell's reciprocal basis. Unwanted segments can be removed from the
written k-path file afterward.

## Step 2: General K Point

The general point is the automatically computed IBZ centroid. The workflow
aborts if it cannot obtain this point; there is no manual-coordinate fallback.

## Step 3: Spin-Flip Operation

Lists the detected spin-flip operations with a default. Accept the default,
pick another by number, or `list` the matrices. Every selectable operation
comes from the detected symmetry set.

## Step 4: Build The General-k Path

The selected operation maps k to k'; both are inserted into the path. Written
coordinates are converted to the submitted reciprocal basis.

## Step 5: Save Outputs

Choose VASP (default), Quantum ESPRESSO, or ABINIT. The workflow writes a fixed
k-path filename and creates or updates the matching plotting configuration.

## Output Files

Files for the selected output code are written to the working directory;
figures and diagnostic records go into `alterseek_output/`.

**Working directory**

| File | Description |
|------|-------------|
| `KPOINTS_alter` | General-k path for VASP line-mode band calculations |
| `KPOINTS_alter_<code>` | QE or ABINIT k-path; `<code>` is `qe` or `abinit` |
| `alterseek_plot_<code>.toml` | Plot configuration; `<code>` is `vasp`, `qe`, or `abinit` |

**`alterseek_output/`**

| File | Description |
|------|-------------|
| `*_ibz_*`, `*_generalpath_*`, `*_spinflip_*`, `*_spinbz_*`, `*_spinbz_top_*` | BZ and IBZ figures; the red-only general-k figure for a non-altermagnet (`*_2d_generalpath_*` in 2D mode); and spin-flip/spin-pattern figures for an altermagnet (`.png`, plus `.pdf` when requested) |
| `spin*_operations.txt` | Full, spin-flip, and spin-preserving operation logs; written only when spin analysis runs |
| `*_magnetic_primitive.mcif` | Direct FindSpinGroup magnetic primitive reference with vector moments; it is not the calculation cell |
| `*_seekpath_basis_mapping.txt` | Submitted analysis/output lattice, internal primitive path lattice, standardized conventional lattice, and rotation mapping |
| `figure_camera_angle.txt` | `view_elev`/`view_azim` each 3D figure was saved at, ready to paste into `alterseek_input.toml` |
| `alterseek_run.log` | Everything a successful run printed; overwritten by the next successful run |

## Save Figures Without Popup Windows

To save the figures without opening interactive windows, select Matplotlib's
non-interactive `Agg` backend before starting the workflow.

**Windows PowerShell:**

```powershell
$env:MPLBACKEND = "Agg"
alterseek-path
```

This setting applies to subsequent commands in the same PowerShell session.
To restore Matplotlib's default backend selection for later runs:

```powershell
Remove-Item Env:MPLBACKEND
```

**Linux/macOS (Bash or Zsh):**

Set the backend once for the current terminal session:

```bash
export MPLBACKEND=Agg
alterseek-path
```

Subsequent runs in the same session need only `alterseek-path`.
To restore Matplotlib's default backend selection for later runs:

```bash
unset MPLBACKEND
```

Alternatively, apply the setting to just one run:

```bash
MPLBACKEND=Agg alterseek-path
```

## Cell Handling

The submitted structure remains the calculation and output cell, including
when it is a conventional cell or supercell. Its reciprocal basis is used for
the written k-points, folded BZ, IBZ, path, and figures. Primitive or
standardized cells used internally never replace it. The
`*_magnetic_primitive.mcif` file is a symmetry reference.

## Input Configuration

If an `alterseek_input.toml` file is present in the working directory,
`alterseek-path` reads the values from it. Any key can be omitted and entered
interactively instead.

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

| Key | Input | Notes |
|-----|-------|-------|
| `structure` | Step 0 structure file | default `POSCAR` if omitted |
| `spin_axis` | Step 0 spin axis | Cartesian axis used by FindSpinGroup to construct collinear spin vectors from scalar moments; it does not fix a physical spin direction in the no-SOC workflow and is ignored for `.mcif` input |
| `moments` | Step 0 magnetic moments | VASP `MAGMOM`-style string; use `""` for no moments |
| `symprec` | symmetry-detection tolerance | positive number in angstrom (default `1e-3`); lower it to preserve smaller distortions |
| `flip_option` | Step 3 spin-flip operation | plain integer, picks that numbered detected operation; omit for the interactive numbered menu (`list` remains available) |
| `output_code` | Step 5 output code | `"vasp"`, `"qe"`, or `"abinit"` |
| `save_pdf` | BZ figure output format | when `true`, also saves each figure as PDF |
| `view_elev`, `view_azim` | 3D camera angle | shared initial angle; current values appear in each interactive 3D figure |
| `vacuum_axis` | 2D-slab vacuum axis of the input cell | `"a"`, `"b"` or `"c"` (default `"c"`); 2D mode only, overridden by an explicit `--vacuum-axis` |

## 2D / Slab Mode

For a slab, `alterseek-path --2d` determines the layer group and 2D lattice
and generates the path, IBZ centroid, and written k-points in the physical 2D
reciprocal plane. Use `--vacuum-axis {a,b,c}` (default `c`) to identify the
submitted cell's vacuum vector.

Instead of the 3D figures, 2D mode saves top-down figures:

| File | Description |
|------|-------------|
| `*_2d_ibz_*.png` | 2D BZ, IBZ, in-plane path, and the general k point |
| `*_2d_spinflip_*.png` | spin-up IBZ and its spin-flip image with path connections |
| `*_2d_spinbz_*.png` | spin-colored 2D BZ domain pattern |

## Next Step

After the band calculation, see [Plotting](plotting.md) for VASP, QE, and
ABINIT spin-resolved band-plot generation.
