import pandapower as pp
import pandapower.networks as nw
import pandas as pd

#Import an example network:
net = nw.case14()
print("line")
print(net.line)
print("bus")
print(net.bus)
print("load")
print(net.load)


print("running the load")
pp.runpp(net)
load_bus_results = pd.merge(net.res_bus, net.load, left_index=True, right_on="bus")
print(load_bus_results)
