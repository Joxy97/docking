from pathlib import Path
from subprocess import CalledProcessError, run


VINA_EXE = Path(r"C:\Program Files (x86)\The Scripps Research Institute\Vina\vina.exe")
OBABEL_EXE = Path(r"C:\Program Files\OpenBabel-3.1.1\obabel.exe")


def leftmost_center(box_size, divisions, center):
	return box_size / 2 * (1 / divisions - 1) + center


def run_checked(command, step_name):
	try:
		run(command, check=True)
	except CalledProcessError as exc:
		raise RuntimeError(f"{step_name} failed with exit code {exc.returncode}") from exc


def guess_element(atom_name):
	letters = "".join(ch for ch in atom_name if ch.isalpha())
	if not letters:
		return "C"
	if len(letters) == 1:
		return letters.upper()
	return letters[0].upper() + letters[1].lower()


def pdbqt_to_pdb(pdbqt_path, pdb_path):
	"""Convert Vina's PDBQT to a PDB that Open Babel can parse."""
	with pdbqt_path.open("r", encoding="utf-8", errors="replace") as src, pdb_path.open(
		"w", encoding="utf-8"
	) as dst:
		for line in src:
			record = line[:6].strip()
			if record in {"MODEL", "ENDMDL", "TER"}:
				dst.write(line)
				continue
			if record not in {"ATOM", "HETATM"}:
				continue
			base = line.rstrip("\n")
			base = base[:66] if len(base) >= 66 else base.ljust(66)
			element = guess_element(line[12:16].strip())
			pdb_line = base.ljust(76) + element.rjust(2) + "\n"
			dst.write(pdb_line)


cx, cy, cz = list(map(float, input("Box center coordinates: ").split()))
bx, by, bz = list(map(float, input("Box size: ").split()))
sx, sy, sz = list(map(int, input("Number of divisions in each direction: ").split()))

x0 = leftmost_center(bx, sx, cx)
y0 = leftmost_center(by, sy, cy)
z0 = leftmost_center(bz, sz, cz)
dx, dy, dz = bx / sx, by / sy, bz / sz

if min(dx, dy, dz) < 3.0:
	print(
		"WARNING: Per-cell box sizes are very small "
		f"({dx:.2f}, {dy:.2f}, {dz:.2f}) and may produce no poses."
	)

centersx = [x0 + i * dx for i in range(sx)]
centersy = [y0 + i * dy for i in range(sy)]
centersz = [z0 + i * dz for i in range(sz)]
centers = [(x, y, z) for x in centersx for y in centersy for z in centersz]

receptor = input("receptor address: ").strip()
ligands = input("ligand address: ").split()
outputs = input("output file: ").split()
docking_config = Path(input("temp config file: ").strip())
temp_output = Path(input("temp output file name: ").strip())

if len(ligands) != len(outputs):
	raise ValueError("The number of ligand files must match the number of output files.")

if not VINA_EXE.exists():
	raise FileNotFoundError(f"Vina executable not found: {VINA_EXE}")
if not OBABEL_EXE.exists():
	raise FileNotFoundError(f"Open Babel executable not found: {OBABEL_EXE}")

for output in outputs:
	Path(output).parent.mkdir(parents=True, exist_ok=True)
	Path(output).write_text("", encoding="utf-8")

temp_pdbqt = temp_output.with_suffix(".pdbqt")
temp_pdb = temp_output.with_suffix(".pdb")
temp_sdf = temp_output.with_suffix(".sdf")

for x, y, z in centers:
	for ligand, output in zip(ligands, outputs):
		print(f"Docking {ligand} and storing in {output}")
		temp_pdbqt.unlink(missing_ok=True)
		temp_pdb.unlink(missing_ok=True)
		temp_sdf.unlink(missing_ok=True)
		with docking_config.open(mode="w", encoding="utf-8") as f:
			f.write(f"receptor = {receptor}\n")
			f.write(f"ligand = {ligand}\n")
			f.write(f"center_x = {x}\n")
			f.write(f"center_y = {y}\n")
			f.write(f"center_z = {z}\n")
			f.write(f"size_x = {dx}\n")
			f.write(f"size_y = {dy}\n")
			f.write(f"size_z = {dz}\n")
			f.write(f"out = {temp_pdbqt}\n")

		run_checked([str(VINA_EXE), "--num_modes", "500", "--config", str(docking_config)], "Vina docking")

		if not temp_pdbqt.exists() or temp_pdbqt.stat().st_size == 0:
			print(
				f"WARNING: Vina produced no poses for {ligand} at center ({x:.2f}, {y:.2f}, {z:.2f})."
			)
			continue

		pdbqt_to_pdb(temp_pdbqt, temp_pdb)
		run_checked([str(OBABEL_EXE), "-ipdb", str(temp_pdb), "-osdf", "-O", str(temp_sdf)], "Open Babel conversion")

		if not temp_sdf.exists() or temp_sdf.stat().st_size == 0:
			print(f"WARNING: Open Babel produced an empty SDF for {ligand}.")
			continue

		with Path(output).open(mode="a", encoding="utf-8") as final_file:
			with temp_sdf.open(mode="r", encoding="utf-8") as temp_file:
				for line in temp_file:
					final_file.write(line)

		temp_pdbqt.unlink(missing_ok=True)
		temp_pdb.unlink(missing_ok=True)
		temp_sdf.unlink(missing_ok=True)
	
	
