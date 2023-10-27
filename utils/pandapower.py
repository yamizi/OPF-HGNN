from torch_geometric.data import HeteroData


def build_costs(net, costs):
     net.gen.cost = 0
     net.ext_grid.cost = 0
     net.sgen.cost = 0

     for cost in costs:
        pp.create_poly_cost(net, cost[1], cost[0], cp1_eur_per_mw=cost[2])
        getattr(net,cost[0]).at[cost[1],'cost']=cost[2]


def build_hetero_data(network):
     
    node_types = ["bus","load","sgen","gen","shunt","ext_grid","line","trafo","trafo3w","impedance","xward"]
    data = HeteroData()
    edges = {}
    dataframes = {}
    for node in node_types:
        edges_ = []
        edges_from = []
        edges_to = []
        edges_to2 = []
        merged_df = getattr(network,node) if len(getattr(network,"res_"+node))==0 else pd.merge(getattr(network,node),getattr(network,"res_"+node),"left",on=None,left_index=True,right_index=True)
        
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

        return data, edges, dataframes