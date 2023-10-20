import warnings
warnings.filterwarnings("ignore")

# imports the pandapower module
import pandapower as pp

# Faulty network

def faulty_example_network():

    net = pp.create_empty_network()

    pp.create_bus(net, name = "110 kV bar", vn_kv = 110, type = 'b', in_service = 'True')
    pp.create_bus(net, name = "20 kV bar", vn_kv = 20, type = 'b')
    pp.create_bus(net, name = "bus 2", vn_kv = 30, type = 'b')
    pp.create_bus(net, name = "bus 3", vn_kv = 20, type = 'b')
    pp.create_bus(net, name = "bus 4", vn_kv = 20, type = 'b')
    pp.create_bus(net, name = "bus 5", vn_kv = -20, type = 'b')
    pp.create_bus(net, name = "bus 6", vn_kv = 20, type = 'b')
    
    pp.create_ext_grid(net, 0, vm_pu = 1)

    pp.create_line(net, name = "line 0", from_bus = 1, to_bus = 2, length_km = 0, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 1", from_bus = 2, to_bus = 3, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 2", from_bus = 3, to_bus = 4, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 3", from_bus = 4, to_bus = 5, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 4", from_bus = 5, to_bus = 6, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 5", from_bus = 6, to_bus = 1, length_km = 1, std_type = "NAYY 4x150 SE")

    pp.create_transformer_from_parameters(net, hv_bus = 1, lv_bus = 0, i0_percent= 0.038, pfe_kw = 11.6,
                                          vkr_percent = 0.322, sn_mva = 40.0, vn_lv_kv = 22.0,
                                          vn_hv_kv = 110.0, vk_percent = 17.8)

    pp.create_load(net, 2, p_mw = -1, q_mvar = 0.200, name = "load 0")
    pp.create_load(net, 3, p_mw = 1, q_mvar = 0.200, name = "load 1")
    pp.create_load(net, 4, p_mw = 1, q_mvar = 0.200, name = "load 2")
    pp.create_load(net, 5, p_mw = 1, q_mvar = 0.200, name = "load 3")
    pp.create_load(net, 6, p_mw = 1, q_mvar = 0.200, name = "load 4")

    pp.create_switch(net, bus = 1, element = 0, et = 'l')
    pp.create_switch(net, bus = 2, element = 0, et = 'l')
    pp.create_switch(net, bus = 2, element = 1, et = 'l')
    pp.create_switch(net, bus = 3, element = 1, et = 'l')
    pp.create_switch(net, bus = 3, element = 2, et = 'l')
    pp.create_switch(net, bus = 4, element = 2, et = 'l')
    pp.create_switch(net, bus = 4, element = 3, et = 'l', closed = False)
    pp.create_switch(net, bus = 5, element = 3, et = 'l')
    pp.create_switch(net, bus = 5, element = 4, et = 'l', closed = False)
    pp.create_switch(net, bus = 6, element = 4, et = 'l', closed = False)
    pp.create_switch(net, bus = 6, element = 5, et = 'l')
    pp.create_switch(net, bus = 1, element = 5, et = 'l')
    
    return net


# creates the example network
faulty_net = faulty_example_network()
# diagnoses the faulty network
errors = pp.diagnostic(faulty_net, report_style="compact")
print(errors)


faulty_net.switch.closed.loc[[8, 9]] = True
faulty_net.bus.vn_kv.loc[[2, 5]] = 20
faulty_net.line.length_km.at[0] = 1
faulty_net.trafo.hv_bus.at[0] = 0
faulty_net.trafo.lv_bus.at[0] = 1
faulty_net.load.p_mw.at[0] = 1

remaining_errors = pp.diagnostic(faulty_net, report_style="compact")
print(remaining_errors)
pp.runpp(faulty_net)
print(faulty_net)

# Overloeaded network
def overload_example_network():

    net = pp.create_empty_network()

    pp.create_bus(net, name = "110 kV bar", vn_kv = 110, type = 'b')
    pp.create_bus(net, name = "20 kV bar", vn_kv = 20, type = 'b')
    pp.create_bus(net, name = "bus 2", vn_kv = 20, type = 'b')
    pp.create_bus(net, name = "bus 3", vn_kv = 20, type = 'b')
    pp.create_bus(net, name = "bus 4", vn_kv = 20, type = 'b')
    pp.create_bus(net, name = "bus 5", vn_kv = 20, type = 'b')
    pp.create_bus(net, name = "bus 6", vn_kv = 20, type = 'b')
    
    pp.create_ext_grid(net, 0, vm_pu = 1)

    pp.create_line(net, name = "line 0", from_bus = 1, to_bus = 2, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 1", from_bus = 2, to_bus = 3, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 2", from_bus = 3, to_bus = 4, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 3", from_bus = 4, to_bus = 5, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 4", from_bus = 5, to_bus = 6, length_km = 1, std_type = "NAYY 4x150 SE")
    pp.create_line(net, name = "line 5", from_bus = 6, to_bus = 1, length_km = 1, std_type = "NAYY 4x150 SE")

    pp.create_transformer_from_parameters(net, hv_bus = 0, lv_bus = 1, i0_percent= 0.038, pfe_kw = 11.6,
        vkr_percent = 0.322, sn_mva = 40.0, vn_lv_kv = 22.0,
        vn_hv_kv = 110.0, vk_percent = 17.8)

    pp.create_load(net, 2, p_mw = 100, q_mvar = 0.2, name = "load 0")
    pp.create_load(net, 3, p_mw = 100, q_mvar = 0.2, name = "load 1")
    pp.create_load(net, 4, p_mw = 100, q_mvar = 0.2, name = "load 2")
    pp.create_load(net, 5, p_mw = 100, q_mvar = 0.2, name = "load 3")
    pp.create_load(net, 6, p_mw = 100, q_mvar = 0.2, name = "load 4")

    pp.create_switch(net, bus = 1, element = 0, et = 'l')
    pp.create_switch(net, bus = 2, element = 0, et = 'l')
    pp.create_switch(net, bus = 2, element = 1, et = 'l')
    pp.create_switch(net, bus = 3, element = 1, et = 'l')
    pp.create_switch(net, bus = 3, element = 2, et = 'l')
    pp.create_switch(net, bus = 4, element = 2, et = 'l')
    pp.create_switch(net, bus = 4, element = 3, et = 'l', closed = 0)
    pp.create_switch(net, bus = 5, element = 3, et = 'l')
    pp.create_switch(net, bus = 5, element = 4, et = 'l')
    pp.create_switch(net, bus = 6, element = 4, et = 'l')
    pp.create_switch(net, bus = 6, element = 5, et = 'l')
    pp.create_switch(net, bus = 1, element = 5, et = 'l')
    
    return net


overload_net = overload_example_network()
try:
    pp.runpp(overload_net)
except Exception as e:
    print(e)
    overload_errors = pp.diagnostic(overload_net, warnings_only = True)
    print(overload_errors)
