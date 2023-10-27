#import the pandapower module$
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
try:
    import seaborn
    colors = seaborn.color_palette()
except:
    colors = ["b", "g", "r", "c", "y"]




def build_advanced_network(net):
    # Double busbar
    pp.create_bus(net, name='Double Busbar 1', vn_kv=380, type='b')
    pp.create_bus(net, name='Double Busbar 2', vn_kv=380, type='b')
    for i in range(10):
        pp.create_bus(net, name='Bus DB T%s' % i, vn_kv=380, type='n')
    for i in range(1, 5):
        pp.create_bus(net, name='Bus DB %s' % i, vn_kv=380, type='n')

    # Single busbar
    pp.create_bus(net, name='Single Busbar', vn_kv=110, type='b')
    for i in range(1, 6):
        pp.create_bus(net, name='Bus SB %s' % i, vn_kv=110, type='n')
    for i in range(1, 6):
        for j in [1, 2]:
            pp.create_bus(net, name='Bus SB T%s.%s' % (i, j), vn_kv=110, type='n')

    # Remaining buses
    for i in range(1, 5):
        pp.create_bus(net, name='Bus HV%s' % i, vn_kv=110, type='n')



    #medium voltage buses
    pp.create_bus(net, name='Bus MV0 20kV', vn_kv=20, type='n')
    for i in range(8):
        pp.create_bus(net, name='Bus MV%s' % i, vn_kv=10, type='n')

    # low voltage buses
    pp.create_bus(net, name='Bus LV0', vn_kv=0.4, type='n')
    for i in range(1, 6):
        pp.create_bus(net, name='Bus LV1.%s' % i, vn_kv=0.4, type='m')
    for i in range(1, 5):
        pp.create_bus(net, name='Bus LV2.%s' % i, vn_kv=0.4, type='m')
    pp.create_bus(net, name='Bus LV2.2.1', vn_kv=0.4, type='m')
    pp.create_bus(net, name='Bus LV2.2.2', vn_kv=0.4, type='m')



    # create lines
    hv_lines = pd.read_csv('example_advanced/hv_lines.csv', sep=';', header=0, decimal=',')
    for _, hv_line in hv_lines.iterrows():
            from_bus = pp.get_element_index(net, "bus", hv_line.from_bus)
            to_bus = pp.get_element_index(net, "bus", hv_line.to_bus)
            pp.create_line(net, from_bus, to_bus, length_km=hv_line.length,std_type=hv_line.std_type, name=hv_line.line_name, parallel=hv_line.parallel)



    mv_lines = pd.read_csv('example_advanced/mv_lines.csv', sep=';', header=0, decimal=',')
    for _, mv_line in mv_lines.iterrows():
        from_bus = pp.get_element_index(net, "bus", mv_line.from_bus)
        to_bus = pp.get_element_index(net, "bus", mv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=mv_line.length, std_type=mv_line.std_type, name=mv_line.line_name)

    lv_lines = pd.read_csv('example_advanced/lv_lines.csv', sep=';', header=0, decimal=',')
    for _, lv_line in lv_lines.iterrows():
        from_bus = pp.get_element_index(net, "bus", lv_line.from_bus)
        to_bus = pp.get_element_index(net, "bus", lv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=lv_line.length, std_type=lv_line.std_type, name=lv_line.line_name)


    # create transformers
    hv_bus = pp.get_element_index(net, "bus", "Bus DB 2")
    lv_bus = pp.get_element_index(net, "bus", "Bus SB 1")
    pp.create_transformer_from_parameters(net, hv_bus, lv_bus, sn_mva=300, vn_hv_kv=380, vn_lv_kv=110, vkr_percent=0.06,
                                        vk_percent=8, pfe_kw=0, i0_percent=0, tp_pos=0, shift_degree=0, name='EHV-HV-Trafo')



    hv_bus = pp.get_element_index(net, "bus", "Bus MV4")
    lv_bus = pp.get_element_index(net, "bus","Bus LV0")
    pp.create_transformer_from_parameters(net, hv_bus, lv_bus, sn_mva=.4, vn_hv_kv=10, vn_lv_kv=0.4, vkr_percent=1.325, vk_percent=4, pfe_kw=0.95, i0_percent=0.2375, tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_step_percent=2.5, tp_pos=0, shift_degree=150, name='MV-LV-Trafo')



    hv_bus = pp.get_element_index(net, "bus", "Bus HV2")
    mv_bus = pp.get_element_index(net, "bus", "Bus MV0 20kV")
    lv_bus = pp.get_element_index(net, "bus", "Bus MV0")
    pp.create_transformer3w_from_parameters(net, hv_bus, mv_bus, lv_bus, vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10, 
                                            sn_hv_mva=40, sn_mv_mva=15, sn_lv_mva=25, vk_hv_percent=10.1, 
                                            vk_mv_percent=10.1, vk_lv_percent=10.1, vkr_hv_percent=0.266667, 
                                            vkr_mv_percent=0.033333, vkr_lv_percent=0.04, pfe_kw=0, i0_percent=0, 
                                            shift_mv_degree=30, shift_lv_degree=30, tap_side="hv", tap_neutral=0, tap_min=-8, 
                                            tap_max=8, tap_step_percent=1.25, tap_pos=0, name='HV-MV-MV-Trafo')

    # create switches
    hv_bus_sw = pd.read_csv('example_advanced/hv_bus_sw.csv', sep=';', header=0, decimal=',')
    # Bus-bus switches
    for _, switch in hv_bus_sw.iterrows():
        from_bus = pp.get_element_index(net, "bus", switch.from_bus)
        to_bus = pp.get_element_index(net, "bus", switch.to_bus)
        pp.create_switch(net, from_bus, to_bus, et=switch.et, closed=switch.closed, type=switch.type, name=switch.bus_name)
    mv_buses = net.bus[(net.bus.vn_kv == 10) | (net.bus.vn_kv == 20)].index
    mv_ls = net.line[(net.line.from_bus.isin(mv_buses)) & (net.line.to_bus.isin(mv_buses))]
    for _, line in mv_ls.iterrows():
            pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
            pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))

    # Bus-line switches
    hv_buses = net.bus[(net.bus.vn_kv == 380) | (net.bus.vn_kv == 110)].index
    hv_ls = net.line[(net.line.from_bus.isin(hv_buses)) & (net.line.to_bus.isin(hv_buses))]
    for _, line in hv_ls.iterrows():
            pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
            pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))

    mv_buses = net.bus[(net.bus.vn_kv == 10) | (net.bus.vn_kv == 20)].index
    mv_ls = net.line[(net.line.from_bus.isin(mv_buses)) & (net.line.to_bus.isin(mv_buses))]
    for _, line in mv_ls.iterrows():
            pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
            pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))

    lv_buses = net.bus[net.bus.vn_kv == 0.4]
    lv_ls = net.line[(net.line.from_bus.isin(lv_buses.index)) & (net.line.to_bus.isin(lv_buses.index))]
    for _, line in lv_ls.iterrows():
            pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
            pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))

    # Trafo-line switches
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus MV4'), pp.get_element_index(net, "trafo", 'MV-LV-Trafo'), et='t', closed=True, type='LBS', name='Switch MV4 - MV-LV-Trafo')
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus LV0'), pp.get_element_index(net, "trafo", 'MV-LV-Trafo'), et='t', closed=True, type='LBS', name='Switch LV0 - MV-LV-Trafo')


    # open switch
    open_switch_id = net.switch[(net.switch.name == 'Switch Bus MV5 - MV Line5')].index
    net.switch.closed.loc[open_switch_id] = False

    # Trafo-line switches
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus DB 2'), pp.get_element_index(net, "trafo", 'EHV-HV-Trafo'), et='t', closed=True, type='LBS', name='Switch DB2 - EHV-HV-Trafo')
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus SB 1'), pp.get_element_index(net, "trafo", 'EHV-HV-Trafo'), et='t', closed=True, type='LBS', name='Switch SB1 - EHV-HV-Trafo')


    # external grid:
    pp.create_ext_grid(net, pp.get_element_index(net, "bus", 'Double Busbar 1'), vm_pu=1.03, va_degree=0, name='External grid',
                    s_sc_max_mva=10000, rx_max=0.1, rx_min=0.1)


    # loads:
    hv_loads = pd.read_csv('example_advanced/hv_loads.csv', sep=';', header=0, decimal=',')
    for _, load in hv_loads.iterrows():
        bus_idx = pp.get_element_index(net, "bus", load.bus)
        pp.create_load(net, bus_idx, p_mw=load.p, q_mvar=load.q, name=load.load_name)

    mv_loads = pd.read_csv('example_advanced/mv_loads.csv', sep=';', header=0, decimal=',')
    for _, load in mv_loads.iterrows():
        bus_idx = pp.get_element_index(net, "bus", load.bus)
        pp.create_load(net, bus_idx, p_mw=load.p, q_mvar=load.q, name=load.load_name)

    lv_loads = pd.read_csv('example_advanced/lv_loads.csv', sep=';', header=0, decimal=',')
    for _, load in lv_loads.iterrows():
        bus_idx = pp.get_element_index(net, "bus", load.bus)
        pp.create_load(net, bus_idx, p_mw=load.p, q_mvar=load.q, name=load.load_name)
        

    # generators
    pp.create_gen(net, pp.get_element_index(net, "bus", 'Bus HV4'), vm_pu=1.03, p_mw=100, name='Gas turbine')

    # static generators
    pp.create_sgen(net, pp.get_element_index(net, "bus", 'Bus SB 5'), p_mw=20, q_mvar=4, sn_mva=45, 
                type='WP', name='Wind Park')

    mv_sgens = pd.read_csv('example_advanced/mv_sgens.csv', sep=';', header=0, decimal=',')
    for _, sgen in mv_sgens.iterrows():
        bus_idx = pp.get_element_index(net, "bus", sgen.bus)
        pp.create_sgen(net, bus_idx, p_mw=sgen.p, q_mvar=sgen.q, sn_mva=sgen.sn, type=sgen.type, name=sgen.sgen_name)

    lv_sgens = pd.read_csv('example_advanced/lv_sgens.csv', sep=';', header=0, decimal=',')
    for _, sgen in lv_sgens.iterrows():
        bus_idx = pp.get_element_index(net, "bus", sgen.bus)
        pp.create_sgen(net, bus_idx, p_mw=sgen.p, q_mvar=sgen.q, sn_mva=sgen.sn, type=sgen.type, name=sgen.sgen_name)


    #shunt
    pp.create_shunt(net, pp.get_element_index(net, "bus", 'Bus HV1'), p_mw=0, q_mvar=0.960, name='Shunt')

    # Impedance
    pp.create_impedance(net, pp.get_element_index(net, "bus", 'Bus HV3'), pp.get_element_index(net, "bus", 'Bus HV1'), 
                        rft_pu=0.074873, xft_pu=0.198872, sn_mva=100, name='Impedance')

    # xwards
    pp.create_xward(net, pp.get_element_index(net, "bus", 'Bus HV3'), ps_mw=23.942, qs_mvar=-12.24187, pz_mw=2.814571, 
                    qz_mvar=0, r_ohm=0, x_ohm=12.18951, vm_pu=1.02616, name='XWard 1')
    pp.create_xward(net, pp.get_element_index(net, "bus", 'Bus HV1'), ps_mw=3.776, qs_mvar=-7.769979, pz_mw=9.174917, 
                    qz_mvar=0, r_ohm=0, x_ohm=50.56217, vm_pu=1.024001, name='XWard 2')




#create an empty network 
network = pp.create_empty_network()

build_advanced_network(network)

pp.runpp(network, calculate_voltage_angles=True, init="dc")
print(network)

bus_df =  pd.merge(network.bus,network.res_bus,"left",on=None,left_index=True,right_index=True)
load_df =  pd.merge(network.load ,network.res_load,"left",on=None,left_index=True,right_index=True)
sgen_df =  pd.merge(network.sgen,network.res_sgen,"left",on=None,left_index=True,right_index=True)
gen_df =  pd.merge(network.gen,network.res_gen,"left",on=None,left_index=True,right_index=True)
shunt_df =  pd.merge(network.shunt,network.res_shunt,"left",on=None,left_index=True,right_index=True)
ext_grid_df =  pd.merge(network.ext_grid,network.res_ext_grid,"left",on=None,left_index=True,right_index=True)
line_df =  pd.merge(network.line,network.res_line,"left",on=None,left_index=True,right_index=True)
trafo_df =  pd.merge(network.trafo,network.res_trafo,"left",on=None,left_index=True,right_index=True)
trafo3w_df =  pd.merge(network.trafo3w,network.res_trafo3w,"left",on=None,left_index=True,right_index=True)
impedance_df =  pd.merge(network.impedance,network.res_impedance,"left",on=None,left_index=True,right_index=True)
xward_df =  pd.merge(network.xward,network.res_xward,"left",on=None,left_index=True,right_index=True)

node_types = ["bus","load","sgen","gen","shunt","ext_grid","line","trafo","trafo3w","impedance","xward"]
edges = {}
dataframes = {}


data = HeteroData()

for node in node_types:
    edges_ = []
    edges_from = []
    edges_to = []
    edges_to2 = []
    merged_df = pd.merge(getattr(network,node),getattr(network,"res_"+node),"left",on=None,left_index=True,right_index=True)
    
    data[node].x = [len(merged_df),len(merged_df.columns)]

    if "from_bus" in merged_df.columns:
        edges_from = merged_df["from_bus"].tolist()
        merged_df.drop(columns=["from_bus"],inplace=True)
        getattr(network,node).drop(columns=["from_bus"],inplace=True)

        edges_to = merged_df["to_bus"].tolist()
        merged_df.drop(columns=["to_bus"],inplace=True)
        getattr(network,node).drop(columns=["to_bus"],inplace=True)

        data['bus','to',node] = [2, len(edges_from)]
        data[node,'to','bus'] = [2, len(edges_to)]

    if "hv_bus" in merged_df.columns:
        edges_from = merged_df["hv_bus"].tolist()
        merged_df.drop(columns=["hv_bus"],inplace=True)
        getattr(network,node).drop(columns=["hv_bus"],inplace=True)
        data['bus','to',node] = [2, len(edges_from)]

        edges_to = merged_df["lv_bus"].tolist()
        merged_df.drop(columns=["lv_bus"],inplace=True)
        getattr(network,node).drop(columns=["lv_bus"],inplace=True)
        data[node,'to','bus'] = [2, len(edges_to)]

    if "mv_bus" in merged_df.columns:
        edges_to2 = merged_df["mv_bus"].tolist()
        merged_df.drop(columns=["mv_bus"],inplace=True)
        getattr(network,node).drop(columns=["mv_bus"],inplace=True)
        data[node,'to','bus'] = [2, len(edges_to)+len(edges_to2)]

    if "bus" in merged_df.columns:
        edges_ = merged_df["bus"].tolist()
        merged_df.drop(columns=["bus"],inplace=True)
        getattr(network,node).drop(columns=["bus"],inplace=True)
        data['bus','to',node] = [2, len(edges_)]

    dataframes[node] = merged_df
    edges[node] = [edges_, edges_from, edges_to, edges_to2]
        

    print(node, len(getattr(network,node).columns), len(merged_df.columns))

print("done")
