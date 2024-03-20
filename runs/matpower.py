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
    mpc.bus = np.array(all_buses)
    octave.push("mpc",mpc)
    octave.eval("[baseMVA, bus, gen, gencost, branch, f, success, et] = runopf(mpc);")
    #octave.eval("[success, results] = runopf(mpc);")
    success = octave.pull("success")
    if not success:
        return None, None
        #octave.eval(f'save("-binary", "converged_{uniqueid}.mat", "results")')

    network = pp.converter.from_ppc(mpc)
    mpc.bus = octave.pull("bus")
    mpc.gen = octave.pull("gen")
    mpc.branch = octave.pull("branch")
    convergence_time = octave.pull("et")

    pp.runpp(network)
    #network.res_bus[["p_mw", "q_mvar"]] = mpc.bus[:, 2:4]
    network.res_bus[["vm_pu","va_degree"]] = mpc.bus[:,7:9]
    generators = mpc.gen[:,0:6]
    ext_grid_bus = network.ext_grid.bus.values[0]
    ext_grid_index = generators[:,0].tolist().index(ext_grid_bus)
    network.res_ext_grid[["p_mw", "q_mvar"]] = generators[ext_grid_index, 1:3]
    network.res_gen[["p_mw", "q_mvar"]] = np.delete(generators,ext_grid_index, axis=0)[:,1:3]
    #network.res_gen[['vm_pu']] = np.delete(generators,ext_grid_index, axis=0)[:,5]

    return network, convergence_time


if __name__ == "__main__":
    case = "case1354pegase"
    #case="case9"
    case_method = getattr(pp.networks, case)
    net = case_method()

    mutation_rate = 0.7
    octave_path = "C:/PortableApps/GNUOctavePortable/App/Octave64/mingw64/bin/octave-cli.exe"
    network, masked_loads = mutate_loads(net, mutation_rate=mutation_rate, relative=True)

    opf(octave_path=octave_path, case=case, loads=masked_loads)