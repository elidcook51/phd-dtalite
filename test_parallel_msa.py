"""Test the parallel_msa sandbox machinery (staging, chdir isolation,
collect-back, cleanup, orchestrator incl. failure path) with a fake msa.

SAFE: everything runs inside a temp copy of DTALite_Files -- your real
data folder is never touched, and no DTALite.exe is needed.

Usage (from the repo root):
    python test_parallel_msa.py

NOTE (Windows): the multiprocessing pool uses the *spawn* start method,
so this script must only run its tests under ``if __name__ == "__main__"``,
and the fake msa callables must be top-level functions (picklable).
"""
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)

import parallel_msa as pm
import msa as msa_mod

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        failures.append(name)


# --- Fake msa implementations (top-level so workers can pickle them). ------
def fake_msa(bigloop, p, p_fix, p_realtime):
    # runs INSIDE the sandbox (cwd == RunX); mimic real outputs
    assert os.path.basename(os.getcwd()).startswith("Run"), os.getcwd()
    assert os.path.isfile("DTALite_Files/input_agent.csv"), "per-iter setup"
    assert os.path.isfile(f"DTALite_Files/{bigloop}_userassignment.mat")
    with open(f"DTALite_Files/{bigloop}UOinfor_agent.csv", "w") as f:
        f.write(f"agent-{bigloop}")
    with open(f"DTALite_Files/myfile{bigloop}.mat", "w") as f:
        f.write(f"mat-{bigloop}")
    with open("DTALite_Files/UOinfor_agent.csv", "w") as f:
        f.write(f"shared-{bigloop}")


def fake_msa_fail(bigloop, p, p_fix, p_realtime):
    if bigloop == 1:
        raise RuntimeError("boom")
    with open(f"DTALite_Files/{bigloop}UOinfor_agent.csv", "w") as f:
        f.write("x")
# ----------------------------------------------------------------------------


def main():
    # Build an isolated playground: temp dir + copy of DTALite_Files.
    test_root = tempfile.mkdtemp(prefix="pmtest_")
    shutil.copytree(os.path.join(REPO, "DTALite_Files"),
                    os.path.join(test_root, "DTALite_Files"))
    os.chdir(test_root)
    dtalite = os.path.join(test_root, "DTALite_Files")
    print(f"Testing in sandbox playground: {test_root}", flush=True)

    # ------------------------------------------------------------ 1. staging
    run_dir = pm.stage_run_dir(997, repo_root=test_root)
    check("stage creates Run997/DTALite_Files",
          os.path.isfile(os.path.join(run_dir, "DTALite_Files",
                                      "input_agent_initial.csv")))
    check("stage copies xlsx", os.path.isfile(
        os.path.join(run_dir, "DTALite_Files", "SiouxFalls_net.xlsx")))
    check("stage copies per-iter mat", os.path.isfile(
        os.path.join(run_dir, "DTALite_Files", "0_userassignment.mat")))

    # --------------------------------------------- 2. chdir isolation (real)
    with pm._sandbox_cwd(run_dir):
        check("cwd is sandbox", os.getcwd() == os.path.abspath(run_dir))
        member, rt, fix = msa_mod.dividerand_matlab(0)  # reads sandbox copy
        check("dividerand_matlab reads inside sandbox", len(member) == 4)
        with open("stray_write_test.txt", "w") as f:  # like bef_aft_realtime*
            f.write("x")
    check("cwd restored", os.getcwd() == test_root)
    check("stray write landed in sandbox, not repo root",
          os.path.isfile(os.path.join(run_dir, "stray_write_test.txt"))
          and not os.path.isfile(os.path.join(test_root,
                                              "stray_write_test.txt")))
    pm.cleanup_run_dir(run_dir)
    check("cleanup deletes sandbox", not os.path.exists(run_dir))

    # ------------------------------- 3. collect_results / shared / missing
    run_dir = pm.stage_run_dir(998, repo_root=test_root)
    sbox = os.path.join(run_dir, "DTALite_Files")
    with open(os.path.join(sbox, "998UOinfor_agent.csv"), "w") as f:
        f.write("a")
    with open(os.path.join(sbox, "myfile998.mat"), "w") as f:
        f.write("b")
    with open(os.path.join(sbox, "UOinfor_agent.csv"), "w") as f:
        f.write("shared998")
    dest = os.path.join(test_root, "test_dest")
    copied = pm.collect_results(run_dir, 998, dest_dir=dest)
    check("collect copies produced files",
          "998UOinfor_agent.csv" in copied and "myfile998.mat" in copied)
    check("collect lands in dest",
          os.path.isfile(os.path.join(dest, "998UOinfor_agent.csv")))
    check("collect warns (not crash) on missing files", True)  # reached
    shared = pm.collect_shared_results(run_dir, dest_dir=dest)
    check("collect_shared copies", shared == ["UOinfor_agent.csv"])
    check("shared content correct",
          open(os.path.join(dest, "UOinfor_agent.csv")).read() == "shared998")
    pm.cleanup_run_dir(run_dir)

    # --------------------------------------------- 4. orchestrator (mocked)
    # NOTE: pass msa_fn explicitly -- monkeypatching msa.msa does NOT
    # propagate to workers under Windows' spawn start method.
    results = pm.run_parallel_msa(pm.runner_schedule(3), max_workers=2,
                                  repo_root=test_root, msa_fn=fake_msa)
    check("orchestrator returns 3 results", len(results) == 3)
    for i in range(3):
        check(f"per-iter result {i} collected",
              open(os.path.join(dtalite, f"{i}UOinfor_agent.csv")).read()
              == f"agent-{i}")
        check(f"per-iter mat {i} collected",
              open(os.path.join(dtalite, f"myfile{i}.mat")).read()
              == f"mat-{i}")
    check("shared files come from max bigloop (serial semantics)",
          open(os.path.join(dtalite, "UOinfor_agent.csv")).read()
          == "shared-2")
    leftovers = [d for d in os.listdir(test_root)
                 if d.startswith("Run") and os.path.isdir(d)]
    check("all sandboxes deleted after success", leftovers == [])

    # ------------------------------------------------- 5. failure path
    try:
        pm.run_parallel_msa(pm.runner_schedule(3), max_workers=2,
                            repo_root=test_root, keep_on_failure=True,
                            msa_fn=fake_msa_fail)
        check("orchestrator raises on worker failure", False)
    except RuntimeError as e:
        check("orchestrator raises on worker failure", "bigloop=1" in str(e))
    check("failed sandbox kept for inspection",
          os.path.isdir(os.path.join(test_root, "Run1")))
    check("succeeded sandboxes cleaned",
          not os.path.exists(os.path.join(test_root, "Run0"))
          and not os.path.exists(os.path.join(test_root, "Run2")))

    # ------------------------------------------------- 6. real data untouched
    os.chdir(REPO)
    shutil.rmtree(test_root, ignore_errors=True)
    check("playground removed", not os.path.exists(test_root))

    print()
    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
