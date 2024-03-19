import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandapower as pp
from utils.pandapower.mutations import mutate_loads
import os
import numpy as np

def opf(case, loads, working_directory="./output",uniqueid="default", octave_path=None):

    # Loads are relative changes

    if octave_path is not None:
        os.environ["OCTAVE_EXECUTABLE"] = octave_path

    from oct2py import Oct2Py, Oct2PyError
    try:
        octave = Oct2Py()
    except Oct2PyError as e:
        print(e)  # noqa
        return None

    matpower_directory = f"{working_directory}/matpower"
    os.makedirs(matpower_directory,exist_ok=True)
    octave.addpath(matpower_directory)
    file_name = f"{matpower_directory}/{uniqueid}.mat"
    converged_name = f"{matpower_directory}/converged_{uniqueid}.mat"

    octave.eval(f"mpc = loadcase('{case}');")
    mpc = octave.pull("mpc")
    buses = mpc.bus

    """
    PQ bus (Loads)        = 1
    PV bus (Generators)   = 2
    """
    all_buses = dict(zip(range(len(buses)), buses.tolist()))
    pq_loads = {a:v for a,v in enumerate(buses.tolist()) if (v[1]==1 and v[2]!=0and v[3]!=0)}

    pq = np.array(list((pq_loads.values())))
    multiplier = np.ones_like(pq)
    multiplier[:len(loads), 2:4] = loads[:, 1:] + 1
    pq_updated = pq * multiplier

    pq_loads_updated = {a:pq_updated[i].tolist() for i, a in enumerate(pq_loads.keys())}
    all_buses = list({**all_buses, **pq_loads_updated}.values())
    mpc.bus = all_buses
    octave.push("mpc",mpc)
    octave.eval("[baseMVA, bus, gen, gencost, branch, f, success, et] = runopf(mpc);")
    #octave.eval("[success, results] = runopf(mpc);")
    success = octave.pull("success")
    if not success:
        return None, None
        #octave.eval(f'save("-binary", "converged_{uniqueid}.mat", "results")')

    mpc.bus = octave.pull("bus")
    mpc.gen = octave.pull("gen")
    mpc.branch = octave.pull("branch")
    convergence_time = octave.pull("et")

    network = pp.converter.from_ppc(mpc)
    pp.runpp(network)
    converged = network.converged
    return network, convergence_time


if __name__ == "__main__":
    case = "case1354pegase"
    #case="case9"
    case_method = getattr(pp.networks, case)
    net = case_method()

    mutation_rate = 1#0.7
    octave_path = "C:/PortableApps/GNUOctavePortable/App/Octave64/mingw64/bin/octave-cli.exe"
    network, masked_loads = mutate_loads(net, mutation_rate=mutation_rate, relative=True)

    opf(octave_path=octave_path, case=case, loads=masked_loads)