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

if os.path.exists(dst):
    os.remove(dst)

shutil.copy(src, dst)

msa.msa(1, 0.1, (1-0.1)*0.6, (1-0.1)*0.4)