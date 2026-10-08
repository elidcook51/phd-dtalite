"""
Post-run UO diagnostic for target agents. FIXED VERSION.
Run from the repo root. Outputs to ./debug_out/uo_diagnosis.csv
"""
import csv
import os
import pandas as pd

TARGET_AGENTS = [5855, 5863, 5874, 5948, 5995]

MATLAB_PATHS = {
    5855: "10;16;17;",
    5863: "12;13;24;21;22;15;",
    5874: "12;11;10;16;18;7;",
    5948: "14;15;19;17;16;8;",
    5995: "20;18;16;10;",
}

def main():
    # Save INSIDE repo, not parent
    os.makedirs("debug_out", exist_ok=True)
    out_path = "debug_out/uo_diagnosis.csv"
    
    print(f"CWD: {os.getcwd()}")
    
    # Load with dtype=str to avoid mixed type issues
    py_uo = pd.read_csv("DTALite_Files/output_agent.csv", usecols=[0, 29], 
                        header=None, dtype=str, low_memory=False)
    print(f"Loaded output_agent.csv: {len(py_uo)} rows")
    
    inp = pd.read_csv("DTALite_Files/input_agent.csv", usecols=[0, 1, 2], 
                      header=None, dtype=str, low_memory=False)
    print(f"Loaded input_agent.csv: {len(inp)} rows")
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_id", "origin", "destination", 
                    "python_path", "matlab_path", "match"])
        
        for aid in TARGET_AGENTS:
            aid_str = str(aid)
            # Compare as strings
            py_row = py_uo[py_uo[0] == aid_str]
            py_path = py_row.iloc[0, 1] if len(py_row) > 0 else "NOT_FOUND"
            
            inp_row = inp[inp[0] == aid_str]
            o = inp_row.iloc[0, 1] if len(inp_row) > 0 else "?"
            d = inp_row.iloc[0, 2] if len(inp_row) > 0 else "?"
            
            matlab_path = MATLAB_PATHS.get(aid, "?")
            # Strip whitespace for comparison
            match = str(py_path).strip() == matlab_path.strip()
            
            w.writerow([aid, o, d, py_path, matlab_path, match])
            print(f"Agent {aid}: OD=({o},{d})")
            print(f"  Python: {py_path}")
            print(f"  MATLAB: {matlab_path}")
            print(f"  Match: {match}")
            print()
    
    print(f"Saved to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
