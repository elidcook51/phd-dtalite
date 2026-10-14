# =====================================================================
# DO NOT RUN -- STALE FILE.
#
# Leftover agent-5855 debug from pre-parity smoke testing. The executable
# body below reads DTALite_Files/0UOinfor_agent.csv just to print agent
# 5855's UO path, and the real 11-iteration MSA loop is entirely commented
# out. Running this file does nothing useful -- and crashes with a cryptic
# error if 0UOinfor_agent.csv is absent.
#
# The live entry points are:
#     wholerunner.py -> runner_parallel.py -> parallel_msa.py      (Test 1)
#     scenarios_ue.py / scenarios_realtime.py + preference_reaction.py
#                                                             (Tests 2-4)
#
# This file is kept (not renamed) for repo history only. If executed, it
# raises RuntimeError immediately instead of failing cryptically.
# =====================================================================

import generate_links
import assignment
import msa
import traveltimecal
import choiceset
import shutil
import os


if __name__ == "__main__":
    raise RuntimeError(
        "runner.py is STALE -- do not run it (leftover agent-5855 debug; "
        "the 11-iteration MSA loop is commented out). Use the live entry "
        "points instead: wholerunner.py -> runner_parallel.py -> "
        "parallel_msa.py, or scenarios_ue.py / scenarios_realtime.py + "
        "preference_reaction.py."
    )


# ---------------------------------------------------------------------
# Original leftover agent-5855 debug body, kept commented for history.
# ---------------------------------------------------------------------

# generate_links.generate_simple_siouxfalls(mode = 'full')

# Check what tt values were computed for 5855's OD
# This will tell us if the tt array was populated correctly
# import pandas as pd
# uo = pd.read_csv('DTALite_Files/0UOinfor_agent.csv')
# row = uo[uo['agent_id'].astype(str)=='5855']
# print("5855 UO path:", row.iloc[0]['path_node_sequence'])




# src = "DTALite_Files/input_agent_initial.csv"
# dst = "DTALite_Files/input_agent.csv"

# files_to_delete = [
#     "input_agent.csv",
#     "output_agent.csv",
#     "output_day_to_day_MOE.csv",
#     "output_LinkMOE.csv",
#     "output_LinkTDMOE.csv",
#     "output_NetworkTDMOE.csv",
#     "output_ODMOE.csv",
#     "output_ODTDMOE.csv",
#     "output_summary.csv",
#     "output_trip.csv",
#     "iteration_LinkTDMOE.csv",
# ]

# files_to_delete = [f"DTALite_Files/{file}" for file in files_to_delete]

# for i in range(11):
#     if i == 0:
#         p = 0.001
#         p_realtime = 0.4
#         p_fix = 0.6

#     else:
#         p = 0.1 * i

#         p_fix = (1 - p) * 0.6
#         p_realtime = (1 - p) * 0.4

#     if os.path.exists(dst):
#         os.remove(dst)

#     shutil.copyfile(src, dst)

#     msa.msa(i, p, p_fix, p_realtime)

#     for file in files_to_delete:
#         if os.path.exists(file):
#             os.remove(file)