import generate_links
import assignment
import msa
import traveltimecal
import choiceset
import shutil
import os

# generate_links.generate_simple_siouxfalls(mode = 'full')

src = "DTALite_Files/input_agent_initial.csv"
dst = "DTALite_Files/input_agent.csv"

files_to_delete = [
    "input_agent.csv",
    "output_agent.csv",
    "output_day_to_day_MOE.csv",
    "output_LinkMOE.csv",
    "output_LinkTDMOE.csv",
    "output_NetworkTDMOE.csv",
    "output_ODMOE.csv",
    "output_ODTDMOE.csv",
    "output_summary.csv",
    "output_trip.csv",
    "iteration_LinkTDMOE.csv",
]

files_to_delete = [f"DTALite_Files/{file}" for file in files_to_delete]

for i in range(11):
    if i == 0:
        p = 0.001
        p_realtime = 0.4
        p_fix = 0.6

    else:
        p = 0.1 * i

        p_fix = (1 - p) * 0.6
        p_realtime = (1 - p) * 0.4

    if os.path.exists(dst):
        os.remove(dst)

    shutil.copyfile(src, dst)

    msa.msa(i, p, p_fix, p_realtime)

    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)