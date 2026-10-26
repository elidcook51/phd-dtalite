# Run one DTALite loading pass over an input directory, mirroring the exe
# flow: settings -> scenario -> input files -> iteration-0 path assignment
# (agent-based driver) -> network loading -> emission post-pass (if any)
# -> the three output files.
import os
import sys

from .cpp import ProgramStop
from .loading import Simulator
from .network import World
from .outputs import write_agent_files, write_link_moe, write_link_tdmoe


def run_loading(input_dir, output_dir):
    world = World(input_dir)
    if world.flow_model not in (1, 2, 3):
        raise ProgramStop(
            "pysim implements the queue-based flow models 1/2/3; "
            "traffic_flow_model=%d (0=BPR, 4=car-following) is out of scope"
            % world.flow_model)
    if world.num_iterations != 0:
        raise ProgramStop(
            "pysim implements a single loading pass; the scenario requests "
            "%d assignment iterations" % (world.num_iterations + 1))
    world.read_inputs()
    sim = Simulator(world)
    sim.assign_paths_iteration_zero()
    sim.network_loading()
    # Emission tier, resolved in World from output_emission_data
    # (see README_TIERS.md). The line below is the run's tier record:
    # a run can never silently be the wrong tier, because the mode and
    # its source are printed here and stored on world.emission_mode.
    print("pysim emission mode: %s (%s)"
          % (world.emission_mode, world.emission_mode_source))
    if world.emission_mode == "approx":
        from .emissions import calculate_emissions
        calculate_emissions(world)
        world.emissions_computed = True
    elif world.emission_mode == "full":
        # Worker A module. Imported lazily so off/approx runs never
        # need it, and a full run without it fails loud here instead
        # of silently falling back to another tier.
        try:
            from .emissions_full import calculate_emissions_full
        except ImportError as exc:
            # Only the module itself missing means "not dropped in";
            # an ImportError inside it is a real bug and re-raises.
            if getattr(exc, "name", "") in ("pysim.emissions_full",
                                            "emissions_full"):
                raise ProgramStop(
                    "emission mode is full but pysim/emissions_full.py "
                    "is not present (worker A module not yet dropped "
                    "in); refusing to fall back to approx or off")
            raise
        calculate_emissions_full(world)
        if not getattr(world, "emissions_computed", False):
            raise ProgramStop(
                "pysim/emissions_full.py did not set "
                "world.emissions_computed=True (contract breach); "
                "refusing to write agent emissions from unset slots")
    os.makedirs(output_dir, exist_ok=True)
    write_agent_files(world,
                      os.path.join(output_dir, "output_agent.csv"),
                      os.path.join(output_dir, "output_trip.csv"))
    write_link_tdmoe(world, os.path.join(output_dir, "output_LinkTDMOE.csv"),
                     day_no=world.num_iterations + 1)
    write_link_moe(world, os.path.join(output_dir, "output_LinkMOE.csv"))
    return world


def main(argv):
    if len(argv) < 2:
        print("usage: python -m pysim.run <input_dir> [output_dir]")
        return 2
    input_dir = argv[1]
    output_dir = argv[2] if len(argv) > 2 else input_dir
    run_loading(input_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
