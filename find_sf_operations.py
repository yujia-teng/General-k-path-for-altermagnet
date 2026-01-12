import numpy as np
import spglib
import spinspg
from ase.io import read
import sys
import os

# --- HELPER 1: Write FULL details for human reading ---
def write_operations_to_file(filename, rotations, translations, spin_rotations, label_info):
    """Writes all spin symmetry operations to a text file."""
    with open(filename, 'w') as f:
        f.write("="*40 + "\n")
        f.write(f"SPIN SYMMETRY LOG\n")
        f.write("="*40 + "\n\n")
        f.write(f"{label_info}\n\n")
        f.write(f"Total Symmetry Operations: {len(rotations)}\n")
        f.write("-" * 40 + "\n")
        for i in range(len(rotations)):
            f.write(f"Operation {i+1}:\n")
            f.write(f"  Rotation:\n{rotations[i]}\n")
            f.write(f"  Translation:\n{translations[i]}\n")
            f.write(f"  Spin Rotation:\n{spin_rotations[i]}\n")
            f.write("-" * 20 + "\n")
    print(f"[INFO] All operations written to '{filename}'")

# --- HELPER 2: Write ONLY Flip Operations for automation ---
def write_flip_ops_to_file(filename, rotations, spin_rotations):
    """
    Filters operations where Spin Rotation is a flip (det approx -1).
    Writes ONLY the spatial rotation matrices for the K-path generator.
    """
    flip_ops = []
    flip_indices = []
    
    for i, s_rot in enumerate(spin_rotations):
        # Check if determinant is approx -1 (Spin Flip)
        if np.isclose(np.linalg.det(s_rot), -1):
            flip_ops.append(rotations[i])
            flip_indices.append(i + 1)
            
    if not flip_ops:
        print("\n[WARNING] No spin-flipping operations found! File not created.")
        return

    with open(filename, 'w') as f:
        f.write(f"# Found {len(flip_ops)} spin-flipping operations\n")
        f.write(f"# Original Indices: {flip_indices}\n")
        for i, rot in enumerate(flip_ops):
            f.write(f"Operation_{i+1}\n")
            # Write matrix row by row
            for row in rot:
                f.write(f"{row[0]} {row[1]} {row[2]}\n")
            f.write("\n")
            
    print(f"[INFO] {len(flip_ops)} spin-flipping matrices written to '{filename}'")

# ==========================================
# MAIN SCRIPT
# ==========================================

# 1. Structure Loading (Renamed to 'structure')
print("="*40)
print("1. Structure Loading")
print("="*40)

filename = input("Enter structure file name (default: POSCAR): ").strip()
if not filename:
    filename = "POSCAR"

try:
    if not os.path.exists(filename):
        raise FileNotFoundError
    
    # Renamed variable from 'atoms' to 'structure'
    structure = read(filename)
    print(f"Successfully loaded '{filename}' containing {len(structure)} atoms.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Extract data
lattice = structure.get_cell()
positions = structure.get_scaled_positions()
numbers = structure.get_atomic_numbers()
num_atoms = len(structure)

# --- PART 2: Non-Magnetic Space Group (SPG) ---
print("\n" + "="*40)
print("2. Non-Magnetic Space Group Analysis")
print("="*40)

cell = (lattice, positions, numbers)
dataset = spglib.get_symmetry_dataset(cell)

non_mag_label = "Unknown"
if dataset:
    non_mag_label = f"{dataset.international} ({dataset.number})"
    print(f"Space Group: {non_mag_label}")
else:
    print("Non-magnetic symmetry detection failed.")

# --- PART 3: User Input for Magnetism ---
print("\n" + "="*40)
print("3. Magnetic Configuration Input")
print("="*40)
print("Enter magnetic moments (space-separated, e.g., '1 -1'):")

try:
    user_input = input("Moments: ").strip()
    if not user_input:
        user_mags = []
    else:
        user_mags = [float(x) for x in user_input.split()]
except ValueError:
    print("Error: Invalid input. Please enter numbers.")
    sys.exit(1)

if len(user_mags) < num_atoms:
    user_mags.extend([0.0] * (num_atoms - len(user_mags)))
elif len(user_mags) > num_atoms:
    user_mags = user_mags[:num_atoms]

# Format as (N, 3) matrix
magmoms = np.zeros((num_atoms, 3))
for i, m in enumerate(user_mags):
    magmoms[i] = [0, 0, m]

print(f"Using magnetic moments:\n{magmoms}")

# --- PART 4: Spin Space Group (SpinSPG) ---
print("\n" + "="*40)
print("4. Spin Space Group Analysis")
print("="*40)

# Run spinspg
sog, rotations, translations, spin_rotations = spinspg.get_spin_symmetry(
    lattice, positions, numbers, magmoms, symprec=1e-5
)

# Run spglib for Magnetic Space Group Label (if available)
msg_label = "Not found (spglib too old?)"
try:
    # Use spglib to get the standard Magnetic Space Group Label (e.g. P6_3'm'c)
    mag_dataset = spglib.get_magnetic_symmetry_dataset(
        (lattice, positions, numbers), magmoms=user_mags, symprec=1e-5
    )
    if mag_dataset and 'uni_symbol' in mag_dataset:
         msg_label = f"{mag_dataset['uni_symbol']} (MSG No. {mag_dataset['uni_number']})"
    elif mag_dataset and 'international' in mag_dataset:
         msg_label = mag_dataset['international']
except Exception:
    pass

# Print info
print(f"Spin-Only Group Type: {sog}")
print(f"Magnetic Space Group: {msg_label}")
print(f"Total Symmetry Operations: {len(rotations)}")

# --- PART 5: Output Files ---
print("\n" + "="*40)
print("5. Saving Results")
print("="*40)

# Prepare label info for text file
label_info_str = f"""Non-Magnetic Label: {non_mag_label}
Spin-Only Group Type: {sog}
Magnetic Space Group Label: {msg_label}"""

# 1. Write the full readable log with LABELS
write_operations_to_file("spin_operations.txt", rotations, translations, spin_rotations, label_info_str)

# 2. Write the automation file
write_flip_ops_to_file("flip_spin_operations.txt", rotations, spin_rotations)
