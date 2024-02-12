from pandapower.topology.create_graph import create_nxgraph
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import torch
from utils.io import JSONEncoder
from typing import Callable, Optional
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)
from pandapower.auxiliary import pandapowerNet
#from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
from copy import deepcopy

class PandaPowerGraph(InMemoryDataset):
    def __init__(self, network: pandapowerNet, preprocess: Optional[str] = None,
                 transform: Optional[Callable] = None, scale=True,
                 pre_transform: Optional[Callable] = None, device="cpu",
                 include_res: bool = True, opf_as_y: bool = True, hetero=True):

        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'metapath2vec', 'transe']
        super().__init__(None, transform, pre_transform)

        self.device = device
        print("Pandapower device",self.device)
        self.node_types = ["bus", "load", "shunt", "ext_grid", "gen", "sgen", "line", "trafo", "trafo3w", "impedance",
                           "xward"]
        if hetero:
            hetero_data, edges, dataframes, scalers = self.build_hetero_data(network, include_res, opf_as_y,
                                                                             scale=scale)
            self.data, self.slices = hetero_data, None
            self.scalers = scalers
            self.dataframes = dataframes
            self.edges = edges
        else:
            homo_data, dataframes, scalers = self.build_homo_data(network, include_res, opf_as_y, scale=scale)
            self.data, self.slices = homo_data, None
            self.scalers = scalers
            self.dataframes = dataframes

    def num_features(self, node):
        default_features = {"bus":12,"load":10,"shunt":10,"ext_grid":15,"gen":18,"sgen":7,"line":27,
                            "trafo":29,"trafo3w":35,"impedance":8,"xward":10}
        return default_features.get(node,1)
    @property
    def num_outputs(self) -> int:
        return 4  # np.sum([len(self._data[e].y.flatten()) for e in self.output_nodes if hasattr(self._data[e],"y")])

    @property
    def output_nodes(self) -> [str]:
        return [e for e in ["ext_grid", "gen", "sgen"] if hasattr(self._data[e], "y")]

    @property
    def total_output_nodes(self) -> [str]:
        return np.sum([len(self.dataframes.get(e)) for e in ["ext_grid", "gen", "sgen"]])*self.num_outputs

    def export(self, filename="export", format="json", experiment=None):

        if format == "csv":
            for (sheetname, sheet) in self.dataframes.items():
                sheet.to_csv(filename + "_" + sheetname + ".csv")
        elif format == "json":
            with open(filename + "." + format, "w") as f:
                json.dump(self.dataframes, f, cls=JSONEncoder)
            if experiment is not None:
                experiment.log_asset(filename + "." + format, filename + "." + format)
        elif format == "xlsx":
            with open(filename + "." + format, "wb") as f:
                for (sheetname, sheet) in self.dataframes.items():
                    sheet.to_excel(f, sheet_name=sheetname)

    def build_homo_data(self, network, include_res=True, opf_as_y=True, scale=True):
        node = "bus"
        bus_df = deepcopy(getattr(network, node)) if (
                    len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(getattr(network, node),
                                                                                                getattr(network,
                                                                                                        "res_" + node),
                                                                                                "left", on=None,
                                                                                                left_index=True,
                                                                                                right_index=True)
        bus_df.drop(columns=["name"], inplace=True)
        bus_df["n_id"] = bus_df.index
        merged_bus_df = bus_df.copy(True)

        node = "ext_grid"
        ext_grid_df = deepcopy(getattr(network, node)) if (
                    len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(getattr(network, node),
                                                                                                getattr(network,
                                                                                                        "res_" + node),
                                                                                                "left", on=None,
                                                                                                left_index=True,
                                                                                                right_index=True)
        ext_grid_df["has_grid"] = 1
        ext_grid_df.drop(columns=["name"], inplace=True)
        if len(ext_grid_df):
            merged_bus_df = pd.merge(merged_bus_df, ext_grid_df, left_on="n_id", right_on="bus", how="left",
                                     suffixes=("", "_ext_grid"))

        node = "gen"
        gen_df = deepcopy(getattr(network, node)) if (
                    len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(getattr(network, node),
                                                                                                getattr(network,
                                                                                                        "res_" + node),
                                                                                                "left", on=None,
                                                                                                left_index=True,
                                                                                                right_index=True)
        gen_df["has_gen"] = 1
        gen_df.drop(columns=["name"], inplace=True)
        if len(gen_df):
            merged_bus_df = pd.merge(merged_bus_df, gen_df, left_on="n_id", right_on="bus", how="left",
                                     suffixes=("", "_gen"))

        node = "sgen"
        sgen_df = deepcopy(getattr(network, node)) if (
                    len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(getattr(network, node),
                                                                                                getattr(network,
                                                                                                        "res_" + node),
                                                                                                "left", on=None,
                                                                                                left_index=True,
                                                                                                right_index=True)
        sgen_df["has_sgen"] = 1
        sgen_df.drop(columns=["name"], inplace=True)
        if len(sgen_df):
            merged_bus_df = pd.merge(merged_bus_df, sgen_df, left_on="n_id", right_on="bus", how="left",
                                     suffixes=("", "_sgen"))

        merged_bus_df = merged_bus_df.fillna(0)
        cols = [a for a in merged_bus_df.columns if
                (("q_mvar" in a) or ("p_mw" in a)) and not ("min" in a or "max" in a)]
        x = merged_bus_df.drop(columns=cols)

        scaler = None #StandardScaler()
        one_hot = pd.get_dummies(x).dropna(axis=1).values.astype("float32")
        if scale and scaler is not None:
            one_hot = scaler.fit_transform(one_hot)
        # x_dict = dict(zip(range(len(one_hot)), one_hot.tolist()))
        x_dict = dict(zip(range(len(one_hot)), torch.Tensor(one_hot).to(self.device)))
        y = np.concatenate([pd.merge(getattr(network, node)[["bus"]], getattr(network, "res_" + node), "left", on=None,
                                     left_index=True, right_index=True)[["bus", "p_mw", "q_mvar"]].values for node in
                            ["ext_grid", "gen", "sgen"]])
        # y_dict = dict(zip(y[:, 0].astype(int), y[:, 1:3].tolist()))
        y_dict = dict(zip(y[:, 0].astype(int), torch.Tensor(y[:, 1:3]).to(self.device)))
        y_dict_default = dict(zip(list(range(len(bus_df))), [torch.Tensor([np.nan, np.nan]).to(self.device)] * len(bus_df)))
        nxgraph = create_nxgraph(network, multi=False, calc_branch_impedances=True)
        nx.set_node_attributes(nxgraph, x_dict, "x")
        nx.set_node_attributes(nxgraph, {**y_dict_default, **y_dict}, "y")
        graph = from_networkx(nxgraph)

        dataframes = {"bus": x,"ext_grid":ext_grid_df,"gen":gen_df,"sgen":sgen_df}
        return graph, dataframes, {"bus": scaler}

    def build_hetero_data(self, network, include_res=True, opf_as_y=True, scale=False):

        node_types = self.node_types
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

            merged_df = deepcopy(getattr(network, node)) if (
                        len(getattr(network, "res_" + node)) == 0 or not include_res) else pd.merge(
                getattr(network, node), getattr(network, "res_" + node), "left", on=None, left_index=True,
                right_index=True)


            if len(merged_df):
                boundaries = np.nan * np.ones((len(merged_df), 8))
                if node == "ext_grid":
                    y = ["p_mw", "q_mvar"]
                    drop_y = y
                    boundaries[:,0:4] = merged_df[["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]].values
                elif node in ["sgen", "gen"]:
                    y = ["p_mw", "q_mvar", "vm_pu", "va_degree"]
                    drop_y = y
                    boundaries[:,0:4] = merged_df[["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]].values
                elif node in ["bus"]:
                    y = ["p_mw", "q_mvar","vm_pu", "va_degree"]
                    boundaries[:,4:6] = merged_df[["min_vm_pu", "max_vm_pu"]].values
                    drop_y = y
                elif node in ["line"]:
                    y = ["pl_mw", "ql_mvar","i_from_ka", "i_to_ka"]
                    drop_y = y
                    max_lines = merged_df[["max_i_ka"]].values
                    boundaries[:,4:8] = np.concatenate([np.zeros_like(max_lines),max_lines, np.zeros_like(max_lines),max_lines],1)

                #print(self.device)
                #print(torch.cuda.is_available())
                #print(torch.cuda.device_count())
                data[node].boundaries = torch.Tensor(boundaries).to(self.device)
            if opf_as_y:

                if node in ["ext_grid", "gen", "sgen", "bus","line"] and len(getattr(network, "res_" + node)) > 0:
                    merged_df = merged_df.rename(columns={"p_mw_y": "p_mw", "vm_pu_y": "vm_pu"})
                    if include_res:
                        merged_df.drop(columns=drop_y, inplace=True)
                    pf = getattr(network, "res_" + node)[y]
                    values = torch.Tensor(pf.values).to(self.device)
                    # Normalize P & Q with nominal voltage
                    values[:, 0] = values[:, 0] / network.sn_mva
                    values[:, 1] = values[:, 1] / network.sn_mva
                    data[node].y = values

                    if node in ["ext_grid", "gen", "sgen"]:
                        node_cost = costs[costs["et"] == node]
                        node_cost.index = node_cost.element
                        merged_df = pd.merge(merged_df, node_cost, how="left", right_index=True, left_index=True).drop(
                            columns=["et", "element"])
                
            
            merged_df.drop(columns=["name"], inplace=True)
            if node=="bus":
                merged_df.drop(columns=["type","zone"], inplace=True)

            scaler = None # StandardScaler()
            one_hot = pd.get_dummies(merged_df).dropna(axis=1).values.astype("float32")
            if scale and scaler is not None:
                one_hot = scaler.fit_transform(one_hot)
            
            
            data[node].x = torch.Tensor(one_hot).to(self.device) if len(one_hot) else torch.zeros(1,self.num_features(node)).to(self.device)
            scalers[node] = scaler

            if "from_bus" in merged_df.columns:
                edges_from = merged_df["from_bus"].tolist()
                merged_df.drop(columns=["from_bus"], inplace=True)
                data['bus', 'to', node].edge_index = torch.LongTensor([edges_from, merged_df.index.tolist()]).to(self.device)

            if "to_bus" in merged_df.columns:
                edges_to = merged_df["to_bus"].tolist()
                merged_df.drop(columns=["to_bus"], inplace=True)
                data[node, 'to', 'bus'].edge_index = torch.LongTensor([merged_df.index.tolist(), edges_to]).to(self.device)

            if "hv_bus" in merged_df.columns:
                edges_from = merged_df["hv_bus"].tolist()
                merged_df.drop(columns=["hv_bus"], inplace=True)
                data['bus', 'to', node].edge_index = torch.LongTensor([edges_from, merged_df.index.tolist()]).to(self.device)

                edges_to = merged_df["lv_bus"].tolist()
                merged_df.drop(columns=["lv_bus"], inplace=True)
                data[node, 'to', 'bus'].edge_index = torch.LongTensor([merged_df.index.tolist(), edges_to]).to(self.device)

            if "mv_bus" in merged_df.columns:
                edges_to2 = merged_df["mv_bus"].tolist()
                merged_df.drop(columns=["mv_bus"], inplace=True)
                data[node, 'to', 'bus'].edge_index = torch.LongTensor(
                    [merged_df.index.tolist() + merged_df.index.tolist(), edges_to + edges_to2]).to(self.device)

            if "bus" in merged_df.columns:
                edges_ = merged_df["bus"].tolist()
                merged_df.drop(columns=["bus"], inplace=True)
                data['bus', 'to', node].edge_index = torch.LongTensor([edges_, merged_df.index.tolist()]).to(self.device)

            dataframes[node] = merged_df
            edges[node] = [edges_, edges_from, edges_to, edges_to2]

            # node, len(getattr(network,node).columns), len(merged_df.columns))

        return data, edges, dataframes, scalers