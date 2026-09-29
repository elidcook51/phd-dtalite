import os
import numpy as np
import pandas as pd

matlab_folder = r"matlab_files"
python_folder = r"python_files"

TOL = 1e-9

matlab_files = {f for f in os.listdir(matlab_folder) if f.endswith(".csv")}
python_files = {f for f in os.listdir(python_folder) if f.endswith(".csv")}

common_files = sorted(matlab_files & python_files)

for filename in common_files:

    matlab_path = os.path.join(matlab_folder, filename)
    python_path = os.path.join(python_folder, filename)

    print(f"\nChecking {filename}")

    try:
        # Read everything as strings initially
        matlab_df = pd.read_csv(matlab_path, header=None, dtype=str)
        python_df = pd.read_csv(python_path, header=None, dtype=str)

        # Shape check
        if matlab_df.shape != python_df.shape:
            print(f"  Shape mismatch:")
            print(f"    MATLAB: {matlab_df.shape}")
            print(f"    Python: {python_df.shape}")
            continue

        nrows, ncols = matlab_df.shape

        mismatch_count = 0
        largest_numeric_diff = 0

        for col in range(ncols):

            m_col = matlab_df.iloc[:, col]
            p_col = python_df.iloc[:, col]

            # Attempt numeric conversion
            m_num = pd.to_numeric(m_col, errors="coerce")
            p_num = pd.to_numeric(p_col, errors="coerce")

            # Numeric column if every non-empty value converted
            m_is_numeric = m_num.notna().all()
            p_is_numeric = p_num.notna().all()

            if m_is_numeric and p_is_numeric:

                diff = np.abs(m_num.to_numpy() - p_num.to_numpy())

                bad = np.where(diff > TOL)[0]

                if len(bad) > 0:
                    mismatch_count += len(bad)

                    max_diff = diff.max()
                    largest_numeric_diff = max(
                        largest_numeric_diff,
                        max_diff
                    )

                    print(
                        f"  Numeric mismatch in column {col}: "
                        f"{len(bad)} rows differ"
                    )

                    # Print first few mismatches
                    for r in bad[:5]:
                        print(
                            f"    Row {r}: "
                            f"{m_num.iloc[r]} vs {p_num.iloc[r]} "
                            f"(diff={diff[r]})"
                        )

            else:
                # String comparison
                bad = np.where(
                    m_col.fillna("") != p_col.fillna("")
                )[0]

                if len(bad) > 0:
                    mismatch_count += len(bad)

                    print(
                        f"  String mismatch in column {col}: "
                        f"{len(bad)} rows differ"
                    )

                    for r in bad[:5]:
                        print(
                            f"    Row {r}: "
                            f"'{m_col.iloc[r]}' vs '{p_col.iloc[r]}'"
                        )

        if mismatch_count == 0:
            print("  MATCH")
        else:
            print(f"  Total mismatches: {mismatch_count}")
            print(f"  Largest numeric difference: {largest_numeric_diff}")

    except Exception as e:
        print(f"  ERROR: {e}")