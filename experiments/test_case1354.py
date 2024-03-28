import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandapower as pp

# check sanity:
"""
-load > generation p_min
-+/- Vm change
"""
# with pandapower
net = pp.networks.case1354pegase()
net.line.max_i_ka *= 2
try:
    pp.runopp(net)
    print("pandapower converged", net.OPF_converged)
except Exception as e:
    print("pandapower error", e)

# with Julia Mathpower
net = pp.networks.case1354pegase()
try:
    pp.runpm_ac_opf(net, pm_log_level=5)
    print("matpower converged", net.OPF_converged)
except Exception as e:
    print("matpower error", e)
