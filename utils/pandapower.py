from typing import Callable, Optional
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)
from pandapower.auxiliary import pandapowerNet
import pandapower as pp
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import torch_geometric.transforms as T
import json
import uuid
import os
from pandapower.optimal_powerflow import OPFNotConverged

def mutate_loads(network, min_p=0, max_p=0,min_q=0, max_q=0, clip=False, mutation_rate=0.7, relative=False,
                 factor = 2,reactive_weight = 0.2):
    
    #min_p=-0.08, max_p=0.1,min_q=-0.08, max_q=0.1
    

    if len(network.load)==0:
        return network
    
    if min_p == max_p == min_q == max_q ==0:
        if not relative:
            min_p = network.load.p_mw.min() / factor
            max_p = network.load.p_mw.max() / factor
            min_q = network.load.q_mvar.min() / factor
            max_q = network.load.q_mvar.max() / factor

            if clip:
                min_p = max(min_p,network.load.p_mw.min()/factor)
                max_p = min(max_p,network.load.p_mw.max()/factor)

                min_q = max(min_q,network.load.q_mvar.min()/factor)
                max_q = min(max_q,network.load.q_mvar.max()/factor)
                
                print("clipping min and max loads to {} and {}".format(min_p,max_p))
        else:
            min_p = -0.1#-0.08 
            max_p = 0.1
            min_q = -0.1#-0.08
            max_q = 0.1

    loads = [[i, np.random.uniform(min_p,max_p)*factor,np.random.uniform(min_p,max_p)*factor] for i in range(len(network.load)) ]
    
    mask = np.random.choice(len(loads),(int(len(loads)*mutation_rate)),replace=False)
    masked_loads = np.array(loads)[mask]
    
    print("updating loads", masked_loads)
    if relative:
        masked_loads[:,1] = (masked_loads[:,1] + 1) * network.load.loc[masked_loads[:,0].astype(int),"p_mw"]
        masked_loads[:,2] = (masked_loads[:,2]*reactive_weight + 1) * network.load.loc[masked_loads[:,0].astype(int),"q_mvar"]
    

    network.load.loc[masked_loads[:,0].astype(int),"p_mw"] = masked_loads[:,1]
    network.load.loc[masked_loads[:,0].astype(int),"q_mvar"] = masked_loads[:,2]


    return network

def mutate_costs(network, min_cost=10, max_cost=100, clip=False, mutation_rate=0.7):
    if clip and len(network.poly_cost):
        min_cost = max(min_cost,network.poly_cost.cp1_eur_per_mw.min())
        max_cost = min(max_cost,network.poly_cost.cp1_eur_per_mw.max())

        print("clipping min and max costs to {} and {}".format(min_cost,max_cost))
    costs_grids = [("ext_grid",i,{"cp1_eur_per_mw":np.random.randint(min_cost,max_cost)}) for i in range(len(network.ext_grid)) ]
    costs_gen = [("gen",i,{"cp1_eur_per_mw":np.random.randint(min_cost,max_cost)}) for i in range(len(network.gen)) ]
    costs_sgen = [("sgen",i,{"cp1_eur_per_mw":np.random.randint(min_cost,max_cost)}) for i in range(len(network.sgen)) ]

    costs = costs_grids + costs_gen + costs_sgen
    mask = np.random.choice(len(costs),(int(len(costs)*mutation_rate)),replace=False)
    masked_costs = list(np.array(costs)[mask])
    build_costs(network,masked_costs)

    return network
def build_dataset(case="case9", nbsamples=20, dataset_type="y_no_OPF", save_dataframes="./data", opf=True,
                  mutations = ["cost", "load"], mutation_rate=0.7, uniqueid=None):
    print("building dataset with {nbsamples} variants")
    case_method = getattr(pp.networks, case)
    original_network = case_method()
    network = deepcopy(original_network)
    graph = PandaPowerDataset(network)
    uniqueid = uuid.uuid4() if uniqueid is None else uniqueid
    path = "."

    if save_dataframes is not None:
        path = "{}/{}_{}/".format(save_dataframes,case,uniqueid)
        os.makedirs(path, exist_ok=True)
        graph.export(path+"/raw")

    transforms = [T.ToUndirected(merge=True)]
    graphs = []

    if nbsamples==0:
        pp.runopp(network, delta=1e-16)
        graph_y = PandaPowerDataset(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms))
        return [graph_y]
    
    for sample_id in range(nbsamples):
        network = deepcopy(original_network)

        if mutation_rate>0:
            if "cost" in mutations:
                network = mutate_costs(network, mutation_rate=mutation_rate)
            
            if "load" in mutations:
                network = mutate_loads(network, mutation_rate=mutation_rate)

            if "load_relative" in mutations:
                network = mutate_loads(network, mutation_rate=mutation_rate, relative=True)

        try:
            if opf:
                pp.runopp(network, delta=1e-16)
            else:
                pp.runpp(network, delta=1e-16)
        except Exception as e:
            print("error in opf",e)
            continue

        if dataset_type=="y_no_OPF":
            graph_y = PandaPowerDataset(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms))
        elif dataset_type=="y_OPF":
            graph_y = PandaPowerDataset(network,include_res=True,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms))
        if dataset_type=="no_y_OPF":
            graph_y = PandaPowerDataset(network,include_res=True,opf_as_y=False, preprocess='metapath2vec',
                            transform=T.Compose(transforms))
            
        if save_dataframes is not None:
            path = "{}/{}_{}/".format(save_dataframes,case,uniqueid)
            graph.export(path+"/op_{}".format(sample_id))

        graphs.append(PandaPowerDataset(network,preprocess='metapath2vec'))
   
    return graphs, network, path, uniqueid

def build_costs(net, costs):

     for cost in costs:
        et, index, prices = cost
        
        for i, p in prices.items():
            if p is None:
                prices[i] = 0
         
        found = net.poly_cost[(net.poly_cost.element==index) & (net.poly_cost.et==et)]
        if len(net.poly_cost) and len(found)>0:
            print("updating cost of ",et,index,"to",prices)
            net.poly_cost.loc[found.index,list(prices.keys())] = list(prices.values())
        else:
            print("setting new cost of ",et,index,"to",prices)
            pp.create_poly_cost(net, index, et, check=False, **prices)

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)

class PandaPowerDataset(InMemoryDataset):
    def __init__(self, network: pandapowerNet, preprocess: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 include_res:bool=True, opf_as_y:bool=True):
        
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(None, transform, pre_transform)

        hetero_data, edges, dataframes, scalers= build_hetero_data(network, include_res, opf_as_y)
        self.data, self.slices = hetero_data, None
        self.scalers = scalers
        self.dataframes = dataframes
        self.edges = edges

    @property
    def num_outputs(self) -> int:
        return 2 # np.sum([len(self._data[e].y.flatten()) for e in self.output_nodes if hasattr(self._data[e],"y")])
    
    @property
    def output_nodes(self) -> [str]:
        return [e for e in ["ext_grid","sgen","gen"] if hasattr(self._data[e],"y")]
    
    def export(self,filename="export",format="json"):

        if format=="csv":
            for (sheetname, sheet) in self.dataframes.items():
                sheet.to_csv(filename+"_"+sheetname+".csv")
        elif format=="json":
            with open(filename+"."+format,"w") as f:
                json.dump(self.dataframes, f, cls=JSONEncoder)
        elif format=="xlsx":
            with open(filename+"."+format,"wb") as f:
                for (sheetname, sheet) in self.dataframes.items():
                    sheet.to_excel(f,sheet_name=sheetname)

def build_hetero_data(network, include_res=True, opf_as_y=True):
     
    node_types = ["bus","load","sgen","gen","shunt","ext_grid","line","trafo","trafo3w","impedance","xward"]
    costs = network.poly_cost
    data = HeteroData()
    edges = {}
    dataframes = {}
    scalers = {}
    for node in node_types:
        edges_ = []
        edges_from = []
        edges_to = []
        edges_to2 = []

        merged_df = deepcopy(getattr(network,node)) if (len(getattr(network,"res_"+node))==0 or not include_res) else pd.merge(getattr(network,node),getattr(network,"res_"+node),"left",on=None,left_index=True,right_index=True)
        if len(merged_df)==0:
            continue
        if opf_as_y and node in ["gen","sgen","ext_grid"] and len(getattr(network,"res_"+node))>0:
            y = ["p_mw","q_mvar"]
            if include_res:
                if node=="ext_grid":
                    merged_df.drop(columns=["p_mw","q_mvar"],inplace=True)
                if node in ["sgen","gen"]:
                    merged_df.drop(columns=["va_degree","vm_pu_y", "p_mw_y","q_mvar"],inplace=True)
                    # y = ["p_mw","q_mvar", "va_degree"]
            pf = getattr(network,"res_"+node)[y]
            data[node].y =torch.Tensor(pf.values.tolist())

            node_cost = costs[costs["et"]==node]
            node_cost.index = node_cost.element
            pd.merge(merged_df,node_cost,how="left",right_index=True, left_index=True).drop(columns=["et","element"])

        merged_df.drop(columns=["name"],inplace=True)   
        scaler = StandardScaler()
        data[node].x =torch.Tensor(scaler.fit_transform(pd.get_dummies(merged_df).dropna(axis=1)))
        scalers[node] = scaler

        if "from_bus" in merged_df.columns:
            edges_from = merged_df["from_bus"].tolist()
            merged_df.drop(columns=["from_bus"],inplace=True)
            data['bus','to',node].edge_index = torch.LongTensor([edges_from, merged_df.index.tolist()])
 
        if "to_bus" in merged_df.columns:
            edges_to = merged_df["to_bus"].tolist()
            merged_df.drop(columns=["to_bus"],inplace=True)
            data[node,'to','bus'].edge_index = torch.LongTensor([merged_df.index.tolist(),edges_to])

        if "hv_bus" in merged_df.columns:
            edges_from = merged_df["hv_bus"].tolist()
            merged_df.drop(columns=["hv_bus"],inplace=True)
            data['bus','to',node].edge_index = torch.LongTensor([edges_from, merged_df.index.tolist()])

            edges_to = merged_df["lv_bus"].tolist()
            merged_df.drop(columns=["lv_bus"],inplace=True)
            data[node,'to','bus'].edge_index = torch.LongTensor([merged_df.index.tolist(),edges_to])

        if "mv_bus" in merged_df.columns:
            edges_to2 = merged_df["mv_bus"].tolist()
            merged_df.drop(columns=["mv_bus"],inplace=True)
            data[node,'to','bus'].edge_index = torch.LongTensor([merged_df.index.tolist()+merged_df.index.tolist(),edges_to+edges_to2])

        if "bus" in merged_df.columns:
            edges_ = merged_df["bus"].tolist()
            merged_df.drop(columns=["bus"],inplace=True)
            data['bus','to',node].edge_index = torch.LongTensor([edges_, merged_df.index.tolist()])

        dataframes[node] = merged_df
        edges[node] = [edges_, edges_from, edges_to, edges_to2]
            

        #node, len(getattr(network,node).columns), len(merged_df.columns))

    return data, edges, dataframes, scalers