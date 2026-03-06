import argparse
import csv
import json
import math
import os
import re
import time
from datetime import datetime
from getpass import getpass
from itertools import combinations
from pathlib import Path

import dimod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridBQMSampler
from rdkit import Chem


# -------------------------
# Settings (kept as notebook defaults unless overridden by CLI)
# -------------------------
LOG_RUN = False
NUM_FRAGMENTS = 4
NUM_PAIRS = int(NUM_FRAGMENTS * (NUM_FRAGMENTS - 1) / 2)
Q_LINEAR = 0.95
Q_QUADRATIC = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run docking BQM pipeline from main.ipynb as a script.")
    parser.add_argument(
        "-sampler",
        "--sampler",
        choices=["SA", "BQM"],
        default="SA",
        help="Sampler backend: SA (simulated annealing) or BQM (Leap hybrid BQM).",
    )
    parser.add_argument(
        "-lambda--single",
        "--lambda-single",
        type=float,
        default=1.0,
        help="Weight for single-fragment score term.",
    )
    parser.add_argument(
        "-lambda--pair",
        "--lambda-pair",
        type=float,
        default=1.0,
        help="Weight for pairwise-fragment score term.",
    )
    return parser.parse_args()


def setup_run_dirs() -> tuple[Path, Path, Path, Path]:
    runs_dir = Path("./runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    existing = [p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    nums = []
    for name in existing:
        suffix = name[4:]
        if suffix.isdigit():
            nums.append(int(suffix))

    next_num = (max(nums) + 1) if nums else 1
    run_folder = runs_dir / f"run_{next_num:02d}"
    run_folder.mkdir(exist_ok=False)

    data_folder = run_folder / "data"
    qubo_folder = run_folder / "qubo"
    results_folder = run_folder / "results"
    data_folder.mkdir(parents=True, exist_ok=True)
    qubo_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    print(f"Created: {run_folder}")
    return run_folder, data_folder, qubo_folder, results_folder


def var_name(f: str, p: str) -> str:
    return f"x{{{f}, {p}}}"


def build_bqm(
    fragments: dict[int, pd.DataFrame],
    pairs: dict[tuple[int, int], pd.DataFrame],
    lambda_single: float,
    lambda_pair: float,
    hard_penalty: float,
    include_pairs=None,
) -> dimod.BinaryQuadraticModel:
    if include_pairs is None:
        include_pairs = sorted(pairs.keys())
    else:
        include_pairs = [tuple(pair) for pair in include_pairs]

    allowed_pairs = set(pairs.keys())
    if not set(include_pairs).issubset(allowed_pairs):
        raise ValueError(f"include_pairs must be a subset of available pairs: {allowed_pairs}")
    if len(include_pairs) == 0:
        raise ValueError("include_pairs cannot be empty")

    frag_dfs: dict[str, pd.DataFrame] = {}
    for fid, df in fragments.items():
        c = df.copy()
        c["f"] = c["f"].astype(str)
        c["p"] = c["p"].astype(str)
        c["score"] = c["score"].astype(float)
        frag_dfs[str(fid)] = c

    pair_dfs: dict[tuple[int, int], pd.DataFrame] = {}
    for pair_key, df in pairs.items():
        c = df.copy()
        c["f1"] = c["f1"].astype(str)
        c["f2"] = c["f2"].astype(str)
        c["p1"] = c["p1"].astype(str)
        c["p2"] = c["p2"].astype(str)
        c["score"] = c["score"].astype(float)
        pair_dfs[tuple(map(int, pair_key))] = c

    fragment_ids_str = [str(fid) for fid in sorted(fragments.keys())]

    poses: dict[str, list[str]] = {}
    for fid in fragment_ids_str:
        vals = frag_dfs[fid]["p"].unique().tolist()
        poses[fid] = sorted(vals, key=lambda x: int(x) if str(x).isdigit() else str(x))

    s_single: dict[tuple[str, str], float] = {}
    for fid, df in frag_dfs.items():
        for _, r in df.iterrows():
            s_single[(fid, str(r["p"]))] = float(r["score"])

    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    for f in fragment_ids_str:
        for p in poses[f]:
            v = var_name(f, p)
            bqm.add_variable(v, 0.0)
            bqm.add_linear(v, lambda_single * s_single[(f, p)])

    for pair_key in include_pairs:
        pair_key = tuple(pair_key)
        df = pair_dfs[pair_key]
        f1_key, f2_key = map(str, pair_key)
        for _, r in df.iterrows():
            p1 = str(r["p1"])
            p2 = str(r["p2"])
            score = float(r["score"])
            v1 = var_name(f1_key, p1)
            v2 = var_name(f2_key, p2)
            bqm.add_interaction(v1, v2, lambda_pair * score)

    for f in fragment_ids_str:
        frag_vars = [var_name(f, p) for p in poses[f]]
        for v in frag_vars:
            bqm.add_linear(v, -hard_penalty)
        for i in range(len(frag_vars)):
            for j in range(i + 1, len(frag_vars)):
                bqm.add_interaction(frag_vars[i], frag_vars[j], 2.0 * hard_penalty)
        bqm.offset += hard_penalty

    return bqm


def chosen_poses(sample: dict, fragment_ids: list[str]) -> dict[str, str | None]:
    out: dict[str, list[str]] = {f: [] for f in fragment_ids}
    for k, v in sample.items():
        if not v:
            continue
        if not (k.startswith("x{") and k.endswith("}")):
            continue
        inside = k[2:-1]
        f, p = [s.strip() for s in inside.split(",")]
        if f in out:
            out[f].append(p)
    return {f: (vals[0] if len(vals) == 1 else None) for f, vals in out.items()}


def evaluate_solution(
    sel: dict[str, str | None],
    pair_codes,
    fragment_ids: list[str],
    single_index: dict[str, pd.DataFrame],
    pair_index: dict[str, tuple[str, str, pd.DataFrame]],
) -> dict[str, float]:
    single_score = 0.0
    single_raw = 0.0
    for f in fragment_ids:
        p = sel[f]
        row = single_index[f].loc[p]
        single_score += float(row["score"])
        single_raw += float(row["energy"])

    pair_score = 0.0
    pair_raw = 0.0
    for code in pair_codes:
        fa, fb, idx = pair_index[code]
        row = idx.loc[(sel[fa], sel[fb])]
        pair_score += float(row["score"])
        pair_raw += float(row["energy"])

    return {
        "score_single": single_score,
        "score_pair": pair_score,
        "score_total": single_score + pair_score,
        "raw_single": single_raw,
        "raw_pair": pair_raw,
        "raw_total": single_raw + pair_raw,
    }


def extract_poses(index: int, rows: list[dict]) -> tuple[int, int, int, int]:
    sol = rows[index]["solution"]
    i1, i2, i3, i4 = [int(re.findall(r"\d+", s)[1]) for s in sol]
    return i1, i2, i3, i4


def main() -> None:
    args = parse_args()

    sampler_name = args.sampler
    lambda_single = float(args.lambda_single)
    lambda_pair = float(args.lambda_pair)
    hard_penalty = 10 * (lambda_single * NUM_FRAGMENTS + lambda_pair * NUM_PAIRS)

    run_folder = data_folder = qubo_folder = results_folder = None
    if LOG_RUN:
        run_folder, data_folder, qubo_folder, results_folder = setup_run_dirs()

    # Load raw data
    fragments = {i: pd.read_csv(f"data/raw/fragment_{i}_raw.csv") for i in range(1, NUM_FRAGMENTS + 1)}
    pairs = {
        (i, j): pd.read_csv(f"data/raw/pair_{i}_{j}_raw.csv")
        for i, j in combinations(range(1, NUM_FRAGMENTS + 1), 2)
    }

    # Shift energies
    for _, df in fragments.items():
        df["energy_shifted"] = df["energy"] - df["energy"].min()
    for _, df in pairs.items():
        df["energy_shifted"] = df["energy"] - df["energy"].min()

    # Scale linear energies
    for _, df in fragments.items():
        q = df["energy_shifted"].quantile(Q_LINEAR)
        df["score"] = (df["energy_shifted"] / q).clip(upper=1)

    # Scale pairwise energies
    for _, df in pairs.items():
        df["log_energy"] = np.log1p(df["energy_shifted"])
        q = df["log_energy"].quantile(Q_QUADRATIC)
        df["score"] = (df["log_energy"] / q).clip(upper=1)

    # Optional processed-data export
    if LOG_RUN:
        for i, df in fragments.items():
            df.to_csv(data_folder / f"fragment_{i}_processed.csv", index=False)
        for (i, j), df in pairs.items():
            df.to_csv(data_folder / f"pair_{i}_{j}_processed.csv", index=False)

    # Build BQM
    bqm = build_bqm(
        fragments=fragments,
        pairs=pairs,
        lambda_single=lambda_single,
        lambda_pair=lambda_pair,
        hard_penalty=hard_penalty,
    )

    print("BQM variables:", len(bqm.variables))
    print("BQM interactions:", len(bqm.quadratic))
    print("BQM offset:", bqm.offset)

    # QUBO conversion + matrix plot
    Q, offset = bqm.to_qubo()
    vars_order = sorted(bqm.variables)
    var_to_idx = {v: i for i, v in enumerate(vars_order)}
    n = len(vars_order)

    if LOG_RUN and qubo_folder is not None:
        bqm_model_path = qubo_folder / "bqm_model.bqm"
        with bqm.to_file() as bqm_file:
            bqm_file.seek(0)
            with open(bqm_model_path, "wb") as fout:
                fout.write(bqm_file.read())
        print(f"Saved BQM model: {bqm_model_path}")

    Qmat = np.zeros((n, n), dtype=float)
    for (u, v), bias in Q.items():
        i = var_to_idx[u]
        j = var_to_idx[v]
        if i == j:
            Qmat[i, j] += bias
        else:
            Qmat[i, j] += bias / 2
            Qmat[j, i] += bias / 2

    emax = max(lambda_single, lambda_pair)
    qmasked = np.ma.masked_where(Qmat > emax, Qmat)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("black")

    fig_qubo, ax_qubo = plt.subplots(figsize=(10, 8))
    im_qubo = ax_qubo.imshow(qmasked, cmap=cmap, vmin=0, vmax=emax, aspect="auto")
    fig_qubo.colorbar(im_qubo, ax=ax_qubo, label="QUBO coefficient")
    ax_qubo.set_title("QUBO Matrix Heatmap")
    ax_qubo.set_xlabel("Variable")
    ax_qubo.set_ylabel("Variable")
    fig_qubo.tight_layout()
    if LOG_RUN and qubo_folder is not None:
        qubo_png = qubo_folder / "qubo_matrix.png"
        fig_qubo.savefig(qubo_png, dpi=200, bbox_inches="tight")
        print(f"Saved QUBO matrix plot: {qubo_png}")
    # plt.show()

    print(f"Soft-energy bound: {emax}")
    print(f"QUBO offset: {offset}")
    print(f"Matrix shape: {Qmat.shape}")

    # Sampling
    num_reads = 2000
    num_sweeps = 1000

    if sampler_name == "SA":
        sampler = SimulatedAnnealingSampler()
        num_variables = bqm.num_variables
        num_interactions = bqm.num_interactions
        t0 = time.perf_counter()
        sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
        solving_time = time.perf_counter() - t0
        sampleset = sampleset.aggregate()
        print(f"Solving time ({sampler_name}): {solving_time:.3f} s")
    elif sampler_name == "BQM":
        token = os.getenv("DWAVE_API_TOKEN") or getpass("Enter D-Wave API token: ")
        sampler = LeapHybridBQMSampler(token=token)
        num_variables = bqm.num_variables
        num_interactions = bqm.num_interactions
        t0 = time.perf_counter()
        sampleset = sampler.sample(bqm)
        solving_time = time.perf_counter() - t0
        sampleset = sampleset.aggregate()
        print(f"Solving time ({sampler_name}): {solving_time:.3f} s")
    else:
        raise NameError(f"SAMPLER name {sampler_name} is not valid. Please use SA or BQM")

    # Metadata
    if LOG_RUN and run_folder is not None:
        metadata = {
            "run_number": run_folder.name,
            "datetime": datetime.now().isoformat(),
            "problem_parameters": {
                "NUM_FRAGMENTS": NUM_FRAGMENTS,
                "NUM_PAIRS": NUM_PAIRS,
                "Q_LINEAR": Q_LINEAR,
                "Q_QUADRATIC": Q_QUADRATIC,
            },
            "qubo_parameters": {
                "SAMPLER": sampler_name,
                "HARD_PENALTY": hard_penalty,
                "LAMBDA_SINGLE": lambda_single,
                "LAMBDA_PAIR": lambda_pair,
            },
            "solver_parameters": {
                "SAMPLER": sampler_name,
                "NUM_VARIABLES": num_variables,
                "NUM_INTERACTIONS": num_interactions,
                "SOLVING_TIME": solving_time,
            },
        }
        json_path = run_folder / "run_metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print("Saved run metadata:", json_path)

    # Collect chosen poses
    fragment_ids = [str(fid) for fid in sorted(fragments.keys())]

    single_index: dict[str, pd.DataFrame] = {}
    for fid, df in fragments.items():
        c = df.copy()
        c["p"] = c["p"].astype(str)
        single_index[str(fid)] = c.set_index("p")

    pair_index: dict[str, tuple[str, str, pd.DataFrame]] = {}
    for (fa, fb), df in pairs.items():
        c = df.copy()
        c["p1"] = c["p1"].astype(str)
        c["p2"] = c["p2"].astype(str)
        pair_index[f"{fa}{fb}"] = (str(fa), str(fb), c.set_index(["p1", "p2"]))

    active_pairs = tuple(pair_index.keys())
    if not active_pairs:
        raise ValueError("No valid active pair codes found")

    rows = []
    for rec in sampleset.data(["sample", "energy"]):
        sel = chosen_poses(rec.sample, fragment_ids)
        if any(sel[f] is None for f in fragment_ids):
            continue
        xs = [f"x{{{f}, {sel[f]}}}" for f in fragment_ids]
        metrics = evaluate_solution(sel, active_pairs, fragment_ids, single_index, pair_index)
        rows.append(
            {
                "solution": xs,
                "bqm_energy": float(rec.energy),
                **metrics,
            }
        )
        if len(rows) >= 20:
            break

    for i, row in enumerate(rows, start=1):
        print(
            f"{i:02d}. {row['solution']} | "
            f"score={row['score_total']:.2f} | "
            f"raw_energy={row['raw_total']:.2f}"
        )

    if rows:
        score = f"{rows[0]['score_total']:.2f}"
    else:
        score = "NA"
        print("No valid one-hot solutions found in the sample set.")

    if LOG_RUN and rows and results_folder is not None:
        csv_path = results_folder / f"{sampler_name}_lam-{lambda_pair}-{lambda_single}_solutions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "solution", "score", "raw_energy"])
            for i, row in enumerate(rows, start=1):
                writer.writerow([i, row["solution"], f"{row['score_total']:.2f}", f"{row['raw_total']:.2f}"])
        print(f"Wrote CSV to {csv_path}")
    elif LOG_RUN:
        print("Skipped CSV export because no valid one-hot rows were found.")

    # Draw and export combined molecule (top solution)
    if not rows:
        return

    i1, i2, i3, i4 = extract_poses(0, rows)
    with Chem.SDMolSupplier("./data/spdf/fragment_1.sdf") as s1:
        with Chem.SDMolSupplier("./data/spdf/fragment_2.sdf") as s2:
            with Chem.SDMolSupplier("./data/spdf/fragment_3.sdf") as s3:
                with Chem.SDMolSupplier("./data/spdf/fragment_4.sdf") as s4:
                    mol1, mol2, mol3, mol4 = s1[i1], s2[i2], s3[i3], s4[i4]

                    combined = Chem.CombineMols(mol1, mol2)
                    combined = Chem.CombineMols(combined, mol3)
                    combined = Chem.CombineMols(combined, mol4)

                    editable_mol = Chem.EditableMol(combined)
                    editable_mol.AddBond(12, 21, order=Chem.BondType.SINGLE)  # 1 - 2
                    editable_mol.AddBond(9, 22, order=Chem.BondType.SINGLE)   # 1 - 3
                    editable_mol.AddBond(17, 26, order=Chem.BondType.SINGLE)  # 2 - 4

                    combined = editable_mol.GetMol()
                    combined.GetAtomWithIdx(12).SetNumExplicitHs(0)
                    combined.GetAtomWithIdx(9).SetNumExplicitHs(0)
                    combined.GetAtomWithIdx(17).SetNumExplicitHs(0)
                    Chem.SanitizeMol(combined)

                    if LOG_RUN and results_folder is not None:
                        out_path = results_folder / f"{sampler_name}_lam-{lambda_pair}-{lambda_single}_score-{score}.sdf"
                    else:
                        test_dir = Path(f"test_{sampler_name}_nr-{num_reads}_ns-{num_sweeps}")
                        test_dir.mkdir(parents=True, exist_ok=True)
                        out_path = test_dir / f"{sampler_name}_lam-{lambda_pair}-{lambda_single}_score-{score}.sdf"

                    writer = Chem.SDWriter(str(out_path))
                    writer.write(combined)
                    writer.close()
                    print(f"Wrote SDF to {out_path}")


if __name__ == "__main__":
    main()
