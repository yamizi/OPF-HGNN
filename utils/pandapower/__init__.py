from typing import Callable, Optional
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)
from pandapower.auxiliary import pandapowerNet
import pandapower as pp
import torch
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import torch_geometric.transforms as T
import json
import uuid
import os
from pandapower.optimal_powerflow import OPFNotConverged
from utils.io import JSONEncoder
from utils.pandapower.mutations import mutate_costs, mutate_loads

def build_dataset(case="case9", nbsamples=20, dataset_type="y_no_OPF", save_dataframes="./data", opf=True,
                  mutations = ["cost", "load"], mutation_rate=0.7, uniqueid=None, experiment=None,scale=True):
    print("building dataset with {nbsamples} variants")

    case_method = getattr(pp.networks, case)
    original_network = case_method()
    networks = {"original":original_network, "mutants":[]}
    network = deepcopy(original_network)
    graph = PandaPowerDataset(network,scale=scale)
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
                            transform=T.Compose(transforms),scale=scale)
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

        networks["mutants"].append(network)
        if dataset_type=="y_no_OPF":
            graph_y = PandaPowerDataset(network,include_res=False,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms),scale=scale)
        elif dataset_type=="y_OPF":
            graph_y = PandaPowerDataset(network,include_res=True,opf_as_y=True, preprocess='metapath2vec',
                            transform=T.Compose(transforms),scale=scale)
        if dataset_type=="no_y_OPF":
            graph_y = PandaPowerDataset(network,include_res=True,opf_as_y=False, preprocess='metapath2vec',
                            transform=T.Compose(transforms),scale=scale)
            
        if save_dataframes is not None:
            path = "{}/{}_{}/".format(save_dataframes,case,uniqueid)
            graph.export(path+"op_{}".format(sample_id), experiment=experiment)

        graphs.append(PandaPowerDataset(network,preprocess='metapath2vec'))
   
    return graphs, networks, path, uniqueid


class PandaPowerDataset(InMemoryDataset):
    def __init__(self, network: pandapowerNet, preprocess: Optional[str] = None,
                 transform: Optional[Callable] = None,scale=True,
                 pre_transform: Optional[Callable] = None,
                 include_res:bool=True, opf_as_y:bool=True):
        
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(None, transform, pre_transform)

        hetero_data, edges, dataframes, scalers= build_hetero_data(network, include_res, opf_as_y, scale=scale)
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
    
    def export(self,filename="export",format="json", experiment=None):

        if format=="csv":
            for (sheetname, sheet) in self.dataframes.items():
                sheet.to_csv(filename+"_"+sheetname+".csv")
        elif format=="json":
            with open(filename+"."+format,"w") as f:
                json.dump(self.dataframes, f, cls=JSONEncoder)
            if experiment is not None:
                experiment.log_asset(filename+"."+format,filename+"."+format)
        elif format=="xlsx":
            with open(filename+"."+format,"wb") as f:
                for (sheetname, sheet) in self.dataframes.items():
                    sheet.to_excel(f,sheet_name=sheetname)

def build_hetero_data(network, include_res=True, opf_as_y=True, scale=True):
     
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
            merged_df[["min_p_mw","max_p_mw","min_q_mvar","max_q_mvar"]]
            data[node].boundaries = torch.Tensor(merged_df[["min_p_mw","max_p_mw","min_q_mvar","max_q_mvar"]].values)
            data[node].y =torch.Tensor(pf.values)

            node_cost = costs[costs["et"]==node]
            node_cost.index = node_cost.element
            pd.merge(merged_df,node_cost,how="left",right_index=True, left_index=True).drop(columns=["et","element"])

        merged_df.drop(columns=["name"],inplace=True)   
        scaler = StandardScaler()
        one_hot = pd.get_dummies(merged_df).dropna(axis=1)
        if scale:
            one_hot = scaler.fit_transform(one_hot)
        data[node].x =torch.Tensor(one_hot)
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