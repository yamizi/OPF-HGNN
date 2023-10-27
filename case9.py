import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append("../")

from utils.pandapower import build_costs, build_hetero_data, PandaPowerDataset
import pandapower as pp

network = pp.networks.case9()

#build_costs(network, [("ext_grid",0,10),("gen",0,10),("gen",1,10)])
graph = PandaPowerDataset(network)

pp.runopp(network, delta=1e-16)
graph_y = PandaPowerDataset(network,include_res=False,opf_as_y=True)
graph_opp_y = PandaPowerDataset(network,include_res=True,opf_as_y=True)
graph_opp = PandaPowerDataset(network,include_res=True,opf_as_y=False)

print()

