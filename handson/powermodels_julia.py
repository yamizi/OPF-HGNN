import sys
sys.path.append(".")
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd

def main():
    net = nw.case9()
    res = pd.DataFrame(columns=net.bus.index.tolist())
    for i, snmva in enumerate([1, 13, 45, 78, 98, 100]):
        net.sn_mva = snmva
        pp.runpm_ac_opf(net)
        res.loc[i] = net.res_bus.vm_pu.values
    print(res)


    net = nw.case9()
    res_pymodel = pd.DataFrame(columns=net.bus.index.tolist())
    for i, snmva in enumerate([1, 13, 45, 78, 98, 100]):
        net.sn_mva = snmva
        pp.runopp(net)
        res_pymodel.loc[i] = net.res_bus.vm_pu.values
    print(res_pymodel)
if __name__ == '__main__':
    main()