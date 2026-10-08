"""
Post-run UO diagnostic for target agents.
Run AFTER the main Python run, from the repo root.
Outputs to ../debug_out/uo_diagnosis.csv

Usage: python diagnose_uo.py
"""
import csv
import os
import sys
import numpy as np
import pandas as pd

TARGET_AGENTS = [5855, 5863, 5874, 5948, 5995]

# MATLAB expected paths (from mismatch file)
MATLAB_PATHS = {
    5855: "10;16;17;",
    5863: "12;13;24;21;22;15;",
    5874: "12;11;10;16;18;7;",
    5948: "14;15;19;17;16;8;",
    5995: "20;18;16;10;",
}

def main():
    os.makedirs("../debug_out", exist_ok=True)
    out_path = "../debug_out/uo_diagnosis.csv"
    
    # Load Python's UO output (bigloop 0)
    py_uo = pd.read_csv("DTALite_Files/output_agent.csv", usecols=[0, 29], header=None)
    # Col 0 = agent_id, col 29 = path
    
    # Load input agent to get OD
    inp = pd.read_csv("DTALite_Files/input_agent.csv", usecols=[0, 1, 2], header=None)
    # Col 0 = agent_id, col 1 = origin, col 2 = destination (verify!)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_id", "origin", "destination", 
                    "python_path", "matlab_path", "match"])
        
        for aid in TARGET_AGENTS:
            # Get Python's path
            py_row = py_uo[py_uo[0] == aid]
            py_path = py_row.iloc[0, 1] if len(py_row) > 0 else "NOT_FOUND"
            
            # Get OD
            inp_row = inp[inp[0] == aid]
            o = inp_row.iloc[0, 1] if len(inp_row) > 0 else "?"
            d = inp_row.iloc[0, 2] if len(inp_row) > 0 else "?"
            
            matlab_path = MATLAB_PATHS.get(aid, "?")
            match = str(py_path).strip() == matlab_path.strip()
            
            w.writerow([aid, o, d, py_path, matlab_path, match])
            print(f"Agent {aid}: OD=({o},{d})")
            print(f"  Python: {py_path}")
            print(f"  MATLAB: {matlab_path}")
            print(f"  Match: {match}")
            print()
    
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
