#!/usr/bin/env bash
set -euo pipefail

# Optional override, e.g. PYTHON_BIN=python3 ./run_workpoints.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_ROOT="test_SA_nr-2000_ns-1000"
RUNS_PER_POINT=30

mkdir -p "$OUT_ROOT"

fmt_lambda() {
  printf "%.1f" "$1"
}

run_workpoint() {
  local lambda_single="$1"
  local lambda_pair="$2"

  local ls_fmt lp_fmt group_dir
  ls_fmt="$(fmt_lambda "$lambda_single")"
  lp_fmt="$(fmt_lambda "$lambda_pair")"
  group_dir="$OUT_ROOT/lam_single-${ls_fmt}_lam_pair-${lp_fmt}"
  mkdir -p "$group_dir"

  # Start clean for this workpoint
  rm -f "$group_dir"/SA_lam-"${lp_fmt}"-"${ls_fmt}"_score-*.sdf
  rm -f "$group_dir"/scores_sorted.txt

  echo "============================================================"
  echo "Workpoint: lambda_single=${ls_fmt}, lambda_pair=${lp_fmt}"
  echo "============================================================"

  for run_idx in $(seq 1 "$RUNS_PER_POINT"); do
    echo "Run ${run_idx}/${RUNS_PER_POINT}"

    run_output="$("$PYTHON_BIN" main.py \
      -sampler SA \
      -lambda--single "$lambda_single" \
      -lambda--pair "$lambda_pair" 2>&1)"
    echo "$run_output"

    # Preferred: parse path printed by main.py
    local sdf_path
    sdf_path="$(printf "%s\n" "$run_output" | sed -n 's/^Wrote SDF to //p' | tail -n1 | tr -d '\r')"
    sdf_path="${sdf_path//\\//}"

    # Fallback: newest matching file in root output directory
    if [[ -z "$sdf_path" || ! -f "$sdf_path" ]]; then
      sdf_path="$(ls -1t "$OUT_ROOT"/SA_lam-"${lp_fmt}"-"${ls_fmt}"_score-*.sdf 2>/dev/null | head -n1 || true)"
    fi

    if [[ -z "$sdf_path" || ! -f "$sdf_path" ]]; then
      echo "ERROR: Could not find output SDF for run ${run_idx} (lambda_single=${ls_fmt}, lambda_pair=${lp_fmt})." >&2
      exit 1
    fi

    local base_name run_name
    base_name="$(basename "$sdf_path")"
    run_name="${base_name%.sdf}_run-$(printf "%02d" "$run_idx").sdf"
    mv -f "$sdf_path" "$group_dir/$run_name"
  done

  # Rank by score parsed from filename and keep only best (lowest score)
  local ranked_lines=()
  mapfile -t ranked_lines < <(
    for f in "$group_dir"/SA_lam-"${lp_fmt}"-"${ls_fmt}"_score-*_run-*.sdf; do
      [[ -e "$f" ]] || continue
      bn="$(basename "$f")"
      score="$(printf "%s\n" "$bn" | sed -E 's/.*_score-([0-9]+([.][0-9]+)?)_run-[0-9]+\.sdf/\1/')"
      printf "%s\t%s\n" "$score" "$f"
    done | sort -g -k1,1
  )

  if [[ ${#ranked_lines[@]} -eq 0 ]]; then
    echo "WARNING: No scored SDF files found in $group_dir after runs." >&2
    return
  fi

  printf "%s\n" "${ranked_lines[@]}" > "$group_dir/scores_sorted.txt"

  local best_file best_base best_clean
  best_file="$(printf "%s\n" "${ranked_lines[0]}" | cut -f2-)"
  best_base="$(basename "$best_file")"
  best_clean="$(printf "%s\n" "$best_base" | sed -E 's/_run-[0-9]+\.sdf$/.sdf/')"
  mv -f "$best_file" "$group_dir/$best_clean"
  best_file="$group_dir/$best_clean"

  for ((i = 1; i < ${#ranked_lines[@]}; i++)); do
    f="$(printf "%s\n" "${ranked_lines[$i]}" | cut -f2-)"
    rm -f "$f"
  done

  echo "Kept best file: $best_file"
  echo "Saved ranking:  $group_dir/scores_sorted.txt"
}

declare -A seen
declare -a workpoints

# lambda_single fixed at 1; lambda_pair in (1,2,3,4,5)
for lp in 1 2 3 4 5; do
  key="1,${lp}"
  if [[ -z "${seen[$key]+x}" ]]; then
    seen[$key]=1
    workpoints+=("$key")
  fi
done

# lambda_pair fixed at 1; lambda_single in (1,2,3,4,5)
for ls in 1 2 3 4 5; do
  key="${ls},1"
  if [[ -z "${seen[$key]+x}" ]]; then
    seen[$key]=1
    workpoints+=("$key")
  fi
done

echo "Workpoints to execute (${#workpoints[@]} unique):"
for wp in "${workpoints[@]}"; do
  IFS=',' read -r ls lp <<< "$wp"
  echo "  lambda_single=$ls, lambda_pair=$lp"
done

for wp in "${workpoints[@]}"; do
  IFS=',' read -r ls lp <<< "$wp"
  run_workpoint "$ls" "$lp"
done

echo "All workpoints completed."
