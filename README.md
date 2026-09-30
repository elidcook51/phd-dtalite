# Running DTALite on UVA Rivanna (Linux) via Wine

**Bottom line: yes, it works.** I ran your exact `DTALite.exe` (v1.1.0, 64-bit)
under Wine 9.0 on Linux with your actual Sioux Falls data. The full simulation
completed (36,030 vehicles, assignment + simulation + emissions, exit 0) and
produced all expected outputs (`output_LinkTDMOE.csv`, `output_agent.csv`, …).
Two concurrent instances in separate sandboxes produced **byte-identical**
outputs, so the parallel workflow is safe too.

## What I found

- Your exe is stock **DTALite 1.1.0** (Jan 2017), 64-bit. The current public
  source on GitHub is a rewritten version with an incompatible file interface,
  so rebuilding natively is *not* a drop-in option.
- Wine runs it essentially natively (compute-bound C++ — negligible overhead).
- One catch: the exe needs **`mfc140.dll`** (Microsoft Foundation Classes),
  which Wine does not ship. Drop the included `mfc140.dll` next to
  `DTALite.exe` and Wine loads it automatically.
- The trailing-comma CSV quirk is byte-identical under Wine, so the
  `sanitize_dtalite_csv()` fix you already have covers Rivanna runs too.

## Files in this folder

| File | What to do |
|---|---|
| `assignment.py`, `msa.py` | **Replace** your current files (CRLF, 2600/1244 lines + patch). Only change: the 5 `subprocess.run(DTALite.exe)` call sites now use `wine DTALite.exe` automatically on Linux, unchanged on Windows. |
| `mfc140.dll` | Copy into `DTALite_Files/` next to `DTALite.exe`. `parallel_msa` copies the whole folder into each `RunX/` sandbox, so it propagates automatically. (Also present in your own `C:\Windows\System32` if you'd rather use yours.) |
| `dtalite-wine.def` | Apptainer container recipe (Ubuntu 22.04 + Wine + Python stack). |

## Rivanna setup

```bash
# 1. On a Rivanna frontend:
module load apptainer

# 2. Build the container (do this on an interactive compute node if it's slow):
apptainer build dtalite-wine.sif dtalite-wine.def

# 3. Stage your project (all .py files + DTALite_Files/ with exe + mfc140.dll)
#    e.g. in ~/phd-dtalite/
```

## Running (SLURM batch example)

```bash
#!/bin/bash
#SBATCH -A <your-allocation>
#SBATCH -p standard
#SBATCH -c 10
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J dtalite-msa
#SBATCH -o msa-%A.out

module load apptainer
cd ~/phd-dtalite

# Per-job Wine prefix on fast local storage:
export WINEPREFIX=$TMPDIR/wineprefix
export WINEDEBUG=-all

apptainer exec dtalite-wine.sif python3 runner_parallel.py --iterations 10 --workers 8
```

Notes:
- Keep `--workers` a bit below the CPUs you request (`-c 10` → `--workers 8`).
  Each worker runs one Wine+DTALite instance.
- First validate: run **one** iteration on Rivanna and diff
  `output_LinkTDMOE.csv` against the same iteration from your Windows machine.
  The sim seeds from `DTASettings.txt` (`random_seed = 100`), so outputs
  should match closely (floating-point across platforms can differ in the last
  bits — that's expected and fine).

## If you hit trouble

- `wine: command not found` → the container didn't build right; check the
  `%post` apt step succeeded.
- `exit=53` from the exe → `mfc140.dll` isn't next to `DTALite.exe`.
- Wine prefix issues → delete `$WINEPREFIX` and let it regenerate (one
  auto-creates on first run, ~30 s).
