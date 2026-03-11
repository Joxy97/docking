from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from csv import writer
from itertools import combinations
from pathlib import Path
import re

from rdkit import Chem
from rdkit.Chem import AllChem


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_BOND_SPECS = [
    "1-2:13-11",
    "1-3:10-2",
    "2-4:6-3",
]

VINA_RESULT_PATTERN = re.compile(r"VINA RESULT:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)")
SDFS_FOLDER_PATTERN = re.compile(r"^SDFs_(.+)_(.+)$")


def parse_bond_spec(spec):
    # Format: "fragA-fragB:atomA-atomB" (all indices are 1-based)
    try:
        pair_part, atom_part = spec.split(":")
        fa, fb = map(int, pair_part.split("-"))
        aa, ab = map(int, atom_part.split("-"))
    except Exception as exc:
        raise ArgumentTypeError(
            f"Invalid bond spec '{spec}'. Expected format fragA-fragB:atomA-atomB "
            "(e.g., 1-2:13-10)."
        ) from exc

    if fa <= 0 or fb <= 0 or aa <= 0 or ab <= 0:
        raise ArgumentTypeError(f"All indices must be positive in '{spec}'.")
    if fa >= fb:
        raise ArgumentTypeError(f"Fragment IDs must be increasing (fa < fb) in '{spec}'.")
    return (fa, fb, aa, ab)


def build_parser():
    parser = ArgumentParser(
        description=(
            "Create fragment, non-bonding (clash), and bonding raw CSV files from docked "
            "fragment SDFs."
        )
    )
    parser.add_argument(
        "--poses-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing fragment SDF files. If omitted, uses "
            "data/SDFs_{BOX_SIZE}_{DIVISIONS} from --box-size/--divisions."
        ),
    )
    parser.add_argument(
        "--box-size",
        type=str,
        default="16",
        help="Box size token used to derive default poses dir: data/SDFs_{BOX_SIZE}_{DIVISIONS}.",
    )
    parser.add_argument(
        "--divisions",
        type=str,
        default="2",
        help="Division token used to derive default poses dir: data/SDFs_{BOX_SIZE}_{DIVISIONS}.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for raw CSV files. Defaults to data/raw for data/SDFs, "
            "or data/raw_{BOX_SIZE}_{DIVISIONS} for data/SDFs_{BOX_SIZE}_{DIVISIONS}."
        ),
    )
    parser.add_argument(
        "--fragments",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Fragment IDs to process for fragment and clash CSVs.",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=None,
        help="Optional pose cap per fragment for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files.",
    )

    parser.add_argument(
        "--write-fragment-raw",
        action=BooleanOptionalAction,
        default=True,
        help="Write fragment_<id>_raw.csv files.",
    )
    parser.add_argument(
        "--energy-field",
        choices=["minimizedAffinity", "CNNaffinity", "CNNscore", "vina_result"],
        default="minimizedAffinity",
        help="Field used for fragment_<id>_raw.csv energy values.",
    )
    parser.add_argument(
        "--strict-fragment-energy",
        action=BooleanOptionalAction,
        default=True,
        help="Fail if requested fragment energy field is missing in any pose.",
    )

    parser.add_argument(
        "--write-nonbond-raw",
        action=BooleanOptionalAction,
        default=True,
        help="Write clash_<i>_<j>_raw.csv files.",
    )
    parser.add_argument(
        "--write-bond-raw",
        action=BooleanOptionalAction,
        default=True,
        help="Write bond_<i>_<j>_raw.csv files.",
    )
    parser.add_argument(
        "--method",
        choices=["uff", "mmff94s"],
        default="mmff94s",
        help="Force-field method for interaction energies.",
    )

    parser.add_argument(
        "--bond-spec",
        type=parse_bond_spec,
        action="append",
        default=None,
        help="Bond definition fragA-fragB:atomA-atomB (1-based atom indices).",
    )
    parser.add_argument(
        "--remove-cap-hydrogens",
        action=BooleanOptionalAction,
        default=True,
        help="Remove one explicit H on each bond-anchor atom before adding inter-fragment bond.",
    )
    parser.add_argument(
        "--failure-energy",
        type=float,
        default=1e9,
        help="Assigned energy when bonded-pair evaluation fails.",
    )
    return parser


def resolve_poses_dir(path):
    if path.is_absolute():
        return path
    direct = path
    if direct.exists():
        return direct
    relative_to_repo = REPO_ROOT / path
    if relative_to_repo.exists():
        return relative_to_repo
    return direct


def default_raw_dir_from_poses_dir(poses_dir):
    if poses_dir.name == "SDFs":
        return poses_dir.parent / "raw"

    match = SDFS_FOLDER_PATTERN.fullmatch(poses_dir.name)
    if match:
        box_size, divisions = match.groups()
        return poses_dir.parent / f"raw_{box_size}_{divisions}"

    return REPO_ROOT / "data/raw"


def default_poses_dir_from_grid(box_size, divisions):
    return REPO_ROOT / "data" / f"SDFs_{box_size}_{divisions}"


def resolve_fragment_sdf(poses_dir, frag_id):
    candidates = [
        poses_dir / f"fragment_{frag_id}.sdf",
        poses_dir / f"fragment{frag_id}.sdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_poses(sdf_path, max_poses=None):
    poses = []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            continue
        poses.append(mol)
        if max_poses is not None and len(poses) >= max_poses:
            break
    return poses


def extract_fragment_energy(mol, energy_field):
    if energy_field in {"minimizedAffinity", "CNNaffinity", "CNNscore"}:
        if not mol.HasProp(energy_field):
            return None
        return float(mol.GetProp(energy_field))

    # vina_result path
    for prop_name in mol.GetPropNames():
        value = mol.GetProp(prop_name)
        match = VINA_RESULT_PATTERN.search(value)
        if match:
            return float(match.group(1))
    return None


def energy_uff(mol, include_interfragment):
    mol = prepare_mol_for_ff(mol)
    if not AllChem.UFFHasAllMoleculeParams(mol):
        raise ValueError("UFF parameters are missing for one or more atoms.")
    ff = AllChem.UFFGetMoleculeForceField(
        mol,
        confId=0,
        ignoreInterfragInteractions=not include_interfragment,
    )
    if ff is None:
        raise ValueError("Failed to initialize UFF force field.")
    return ff.CalcEnergy()


def energy_mmff94s(mol, include_interfragment):
    mol = prepare_mol_for_ff(mol)
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError("MMFF94s parameters are missing for one or more atoms.")
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if props is None:
        raise ValueError("Failed to initialize MMFF94s properties.")
    ff = AllChem.MMFFGetMoleculeForceField(
        mol,
        props,
        confId=0,
        ignoreInterfragInteractions=not include_interfragment,
    )
    if ff is None:
        raise ValueError("Failed to initialize MMFF94s force field.")
    return ff.CalcEnergy()


def select_energy_fn(method):
    if method == "uff":
        return energy_uff
    return energy_mmff94s


def prepare_mol_for_ff(mol):
    # CombineMols() returns molecules without initialized ring info in some RDKit builds.
    # MMFF/UFF may raise precondition errors unless ring info and property cache are ready.
    prepared = Chem.Mol(mol)
    prepared.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(prepared)
    return prepared


def remove_one_h_neighbor_per_anchor(rwmol, anchor_a, anchor_b):
    to_remove = []
    atom_a = rwmol.GetAtomWithIdx(anchor_a)
    atom_b = rwmol.GetAtomWithIdx(anchor_b)

    for nbr in atom_a.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            to_remove.append(nbr.GetIdx())
            break
    for nbr in atom_b.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            idx = nbr.GetIdx()
            if idx not in to_remove:
                to_remove.append(idx)
            break

    for idx in sorted(to_remove, reverse=True):
        rwmol.RemoveAtom(idx)

    shift_a = sum(1 for idx in to_remove if idx < anchor_a)
    shift_b = sum(1 for idx in to_remove if idx < anchor_b)
    return anchor_a - shift_a, anchor_b - shift_b


def remove_cap_hydrogen_for_fragment(mol, anchor_1based):
    n_atoms = mol.GetNumAtoms()
    if anchor_1based < 1 or anchor_1based > n_atoms:
        raise IndexError(
            f"Anchor atom index {anchor_1based} out of range for molecule with {n_atoms} atoms."
        )

    editable = Chem.RWMol(Chem.Mol(mol))
    anchor_0based = anchor_1based - 1
    anchor_atom = editable.GetAtomWithIdx(anchor_0based)

    for nbr in anchor_atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            editable.RemoveAtom(nbr.GetIdx())
            break

    trimmed = editable.GetMol()
    trimmed.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(trimmed)
    return trimmed


def build_anchor_self_energy_cache(fragment_poses, bond_specs, energy_fn):
    anchors_by_fragment = {}
    for fa, fb, aa, ab in bond_specs:
        anchors_by_fragment.setdefault(fa, set()).add(aa)
        anchors_by_fragment.setdefault(fb, set()).add(ab)

    cache = {}
    for frag_id, anchors in anchors_by_fragment.items():
        poses = fragment_poses[frag_id]
        for anchor in sorted(anchors):
            energies = []
            for mol in poses:
                trimmed = remove_cap_hydrogen_for_fragment(mol, anchor)
                energies.append(energy_fn(trimmed, include_interfragment=False))
            cache[(frag_id, anchor)] = energies
    return cache


def build_bonded_pair_mol(mol_a, mol_b, atom_a_1based, atom_b_1based, remove_cap_hydrogens):
    n_a = mol_a.GetNumAtoms()
    n_b = mol_b.GetNumAtoms()
    if atom_a_1based < 1 or atom_a_1based > n_a:
        raise IndexError(f"Atom index {atom_a_1based} out of range for first molecule ({n_a} atoms).")
    if atom_b_1based < 1 or atom_b_1based > n_b:
        raise IndexError(f"Atom index {atom_b_1based} out of range for second molecule ({n_b} atoms).")

    combined = Chem.CombineMols(mol_a, mol_b)
    editable = Chem.RWMol(combined)

    atom_a_0based = atom_a_1based - 1
    atom_b_0based = n_a + (atom_b_1based - 1)

    if remove_cap_hydrogens:
        atom_a_0based, atom_b_0based = remove_one_h_neighbor_per_anchor(
            editable, atom_a_0based, atom_b_0based
        )

    editable.AddBond(atom_a_0based, atom_b_0based, order=Chem.BondType.SINGLE)
    bonded = editable.GetMol()
    Chem.SanitizeMol(bonded)
    return bonded


def write_fragment_raw(args, raw_dir, fragment_poses):
    for frag_id in sorted(set(args.fragments)):
        out_path = raw_dir / f"fragment_{frag_id}_raw.csv"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing file (use --overwrite): {out_path}")
            continue

        poses = fragment_poses[frag_id]
        rows = []
        for pose_idx, mol in enumerate(poses):
            energy = extract_fragment_energy(mol, args.energy_field)
            if energy is None:
                if args.strict_fragment_energy:
                    raise ValueError(
                        f"Missing field '{args.energy_field}' for fragment {frag_id}, pose {pose_idx}"
                    )
                continue
            rows.append((frag_id, pose_idx, energy))

        if not rows:
            raise ValueError(
                f"No fragment energies extracted for fragment {frag_id} from field '{args.energy_field}'."
            )

        with out_path.open("w", newline="", encoding="utf-8") as handle:
            csv = writer(handle)
            csv.writerow(["f", "p", "energy"])
            for f, p, e in rows:
                csv.writerow([f, p, f"{e:.10f}"])

        print(f"Wrote {out_path} ({len(rows)} rows, field={args.energy_field})")


def write_nonbond_raw(args, raw_dir, fragment_poses, self_energies, energy_fn):
    for f1, f2 in combinations(sorted(set(args.fragments)), 2):
        out_path = raw_dir / f"clash_{f1}_{f2}_raw.csv"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing file (use --overwrite): {out_path}")
            continue

        poses1 = fragment_poses[f1]
        poses2 = fragment_poses[f2]
        energies1 = self_energies[f1]
        energies2 = self_energies[f2]

        done = 0
        total = len(poses1) * len(poses2)
        print(f"Computing clash pair {f1}-{f2}: {len(poses1)} x {len(poses2)} = {total}")

        with out_path.open("w", newline="", encoding="utf-8") as handle:
            csv = writer(handle)
            csv.writerow(["f1", "f2", "p1", "p2", "energy"])

            for p1, mol1 in enumerate(poses1):
                for p2, mol2 in enumerate(poses2):
                    combined = Chem.CombineMols(mol1, mol2)
                    total_energy = energy_fn(combined, include_interfragment=True)
                    interaction = total_energy - energies1[p1] - energies2[p2]
                    csv.writerow([f1, f2, p1, p2, f"{interaction:.10f}"])
                    done += 1

        print(f"Wrote {out_path} ({done} rows)")


def write_bond_raw(
    args,
    raw_dir,
    bond_specs,
    fragment_poses,
    self_energies,
    anchor_self_energies,
    energy_fn,
):
    for fa, fb, aa, ab in bond_specs:
        out_path = raw_dir / f"bond_{fa}_{fb}_raw.csv"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing file (use --overwrite): {out_path}")
            continue

        poses_a = fragment_poses[fa]
        poses_b = fragment_poses[fb]
        if args.remove_cap_hydrogens:
            energies_a = anchor_self_energies[(fa, aa)]
            energies_b = anchor_self_energies[(fb, ab)]
        else:
            energies_a = self_energies[fa]
            energies_b = self_energies[fb]

        done = 0
        failures = 0
        total = len(poses_a) * len(poses_b)
        print(
            f"Computing bond pair {fa}-{fb} with atoms {aa}-{ab}: "
            f"{len(poses_a)} x {len(poses_b)} = {total}"
        )

        with out_path.open("w", newline="", encoding="utf-8") as handle:
            csv = writer(handle)
            csv.writerow(["f1", "f2", "p1", "p2", "energy"])

            for p1, mol_a in enumerate(poses_a):
                for p2, mol_b in enumerate(poses_b):
                    try:
                        bonded = build_bonded_pair_mol(
                            mol_a,
                            mol_b,
                            aa,
                            ab,
                            remove_cap_hydrogens=args.remove_cap_hydrogens,
                        )
                        total_energy = energy_fn(bonded, include_interfragment=True)
                        interaction = total_energy - energies_a[p1] - energies_b[p2]
                    except Exception:
                        interaction = args.failure_energy
                        failures += 1

                    csv.writerow([fa, fb, p1, p2, f"{interaction:.10f}"])
                    done += 1

        print(f"Wrote {out_path} ({done} rows, failures={failures})")


def main():
    args = build_parser().parse_args()

    if args.poses_dir is None:
        poses_dir = default_poses_dir_from_grid(args.box_size, args.divisions)
    else:
        poses_dir = resolve_poses_dir(args.poses_dir)
    if args.raw_dir is None:
        raw_dir = default_raw_dir_from_poses_dir(poses_dir)
    else:
        raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir

    if not poses_dir.exists():
        raise FileNotFoundError(f"Poses directory not found: {poses_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using poses dir: {poses_dir}")
    print(f"Using raw dir:   {raw_dir}")

    bond_specs = args.bond_spec
    if bond_specs is None:
        bond_specs = [parse_bond_spec(spec) for spec in DEFAULT_BOND_SPECS]

    load_fragments = set(args.fragments)
    if args.write_bond_raw:
        for fa, fb, _, _ in bond_specs:
            load_fragments.add(fa)
            load_fragments.add(fb)

    fragment_poses = {}
    for frag_id in sorted(load_fragments):
        sdf_path = resolve_fragment_sdf(poses_dir, frag_id)
        if not sdf_path.exists():
            raise FileNotFoundError(f"Missing SDF file for fragment {frag_id}: {sdf_path}")

        poses = load_poses(sdf_path, max_poses=args.max_poses)
        if not poses:
            raise ValueError(f"No valid poses found in {sdf_path}")
        fragment_poses[frag_id] = poses
        print(f"Loaded fragment {frag_id} from {sdf_path} ({len(poses)} poses)")

    if args.write_fragment_raw:
        write_fragment_raw(args, raw_dir, fragment_poses)

    if args.write_nonbond_raw or args.write_bond_raw:
        energy_fn = select_energy_fn(args.method)
        self_energies = {
            frag_id: [energy_fn(mol, include_interfragment=False) for mol in poses]
            for frag_id, poses in fragment_poses.items()
        }
        if args.write_bond_raw and args.remove_cap_hydrogens:
            anchor_self_energies = build_anchor_self_energy_cache(
                fragment_poses, bond_specs, energy_fn
            )
        else:
            anchor_self_energies = {}
    else:
        energy_fn = None
        self_energies = {}
        anchor_self_energies = {}

    if args.write_nonbond_raw:
        write_nonbond_raw(args, raw_dir, fragment_poses, self_energies, energy_fn)

    if args.write_bond_raw:
        write_bond_raw(
            args,
            raw_dir,
            bond_specs,
            fragment_poses,
            self_energies,
            anchor_self_energies,
            energy_fn,
        )

    print("Done.")


if __name__ == "__main__":
    main()
