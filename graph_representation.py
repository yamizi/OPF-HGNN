import pandapower as pp
import pandapower.networks as nw
import pandas as pd

from pandapower.topology.create_graph import create_nxgraph
from pandapower.plotting import simple_plot
from matplotlib import pyplot as plt
import networkx as nx

#Import an example network:
net = nw.case14()
pp.runpp(net)

G = create_nxgraph(net)

#Print the number of nodes and edges in the graph:     

print("Number of nodes: ", len(G.nodes))    

print("Number of edges: ", len(G.edges))

#Print the nodes and edges in the graph:

print("Nodes: ", G.nodes)

print("Edges: ", G.edges)

print("Graph: ", G.graph)

simple_graph = nx.Graph(G)
nx.draw(simple_graph, pos=nx.planar_layout(simple_graph))
plt.show()
"""
"""
edges = {"line":"red","trafo":"green"}
#simple_plot(net, plot_loads=True)


pos = nx.planar_layout(G)
names = {name: name for name in G.nodes}
nx.draw_networkx_nodes(G, pos, node_color = 'b', node_size = 250, alpha = 1)
nx.draw_networkx_labels(G,pos,names,font_size=12,font_color='w')
ax = plt.gca()
for e in G.edges:
    ax.annotate("",
                xy=pos[e[1]], xycoords='data',
                xytext=pos[e[0]], textcoords='data',
                arrowprops=dict(arrowstyle="->", color=edges.get(e[2][0]),
                                shrinkA=10, shrinkB=10,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=0".replace('rrr',str(edges.get(e[2][0]))
                                ),
                                ),
                )
plt.axis('off')
plt.show()
