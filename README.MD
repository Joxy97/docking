# Docking Pipeline (GNINA -> CSV -> Main)

## 1) Environment
```bash
cd "path_to_folder"/docking
source .venv/bin/activate
```

## 2) Run GNINA docking (writes SDFs to `data/SDFs`, overwriting)
```bash
cd "path_to_folder"/docking
python3 docking_tools/gnina_grid_docking.py < docking_tools/gnina_docking_input.txt
```

## 3) Export CSV raw files (`fragment_*`, `clash_*`, `bond_*`)
```bash
cd "path_to_folder"/docking
python3 docking_tools/create_csv_files.py
```

## 4) Run main pipeline
```bash
cd "path_to_folder"/docking
python3 main.py
```

## Optional: common main.py flags
```bash
python3 main.py --sampler SA --num-reads 2000 --num-sweeps 1000
python3 main.py --bonding-fragments 1-2,1-3,2-4
python3 main.py --log-run
```
