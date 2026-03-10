from pathlib import Path
from shutil import which
from subprocess import CalledProcessError, run


NUM_MODES = 100
EXHAUSTIVENESS = 64
# Set to True for pure empirical Vina-like scoring instead of GNINA default CNN rescoring.
EMPIRICAL_VINA_ONLY = False


def leftmost_center(box_size, divisions, center):
    return box_size / 2 * (1 / divisions - 1) + center


def run_checked(command, step_name):
    try:
        run(command, check=True)
    except CalledProcessError as exc:
        raise RuntimeError(f"{step_name} failed with exit code {exc.returncode}") from exc


def gnina_executable():
    exe = which("gnina")
    if exe:
        return exe
    raise FileNotFoundError(
        "GNINA executable not found in PATH. Install gnina and ensure `gnina` is available."
    )


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
        f"({dx:.2f}, {dy:.2f}, {dz:.2f}) and may produce unstable or no poses."
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

gnina = gnina_executable()

for output in outputs:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("", encoding="utf-8")

temp_sdf = temp_output.with_suffix(".sdf")
temp_log = temp_output.with_suffix(".log")

for x, y, z in centers:
    for ligand, output in zip(ligands, outputs):
        print("=" * 80)
        print(f"\nDocking {ligand} \nand storing in {output}")
        temp_sdf.unlink(missing_ok=True)
        temp_log.unlink(missing_ok=True)

        with docking_config.open(mode="w", encoding="utf-8") as f:
            f.write(f"receptor = {receptor}\n")
            f.write(f"ligand = {ligand}\n")
            f.write(f"center_x = {x}\n")
            f.write(f"center_y = {y}\n")
            f.write(f"center_z = {z}\n")
            f.write(f"size_x = {dx}\n")
            f.write(f"size_y = {dy}\n")
            f.write(f"size_z = {dz}\n")
            f.write(f"out = {temp_sdf}\n")
            f.write(f"num_modes = {NUM_MODES}\n")
            f.write(f"exhaustiveness = {EXHAUSTIVENESS}\n")

        command = [gnina, "--config", str(docking_config), "--log", str(temp_log)]
        if EMPIRICAL_VINA_ONLY:
            command.extend(["--scoring", "vina", "--cnn_scoring", "none"])

        run_checked(command, "GNINA docking")

        if not temp_sdf.exists() or temp_sdf.stat().st_size == 0:
            print(
                f"WARNING: GNINA produced no poses for {ligand} at center ({x:.2f}, {y:.2f}, {z:.2f})."
            )
            continue

        with Path(output).open(mode="a", encoding="utf-8") as final_file:
            with temp_sdf.open(mode="r", encoding="utf-8", errors="replace") as temp_file:
                for line in temp_file:
                    final_file.write(line)

        temp_sdf.unlink(missing_ok=True)
        temp_log.unlink(missing_ok=True)
