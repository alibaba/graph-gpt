import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.utils import to_networkx

from visualize import GraphVisualization


def get_node_txt(graph: torch_geometric.data.data.Data):
    if hasattr(graph, "id"):
        id_ = graph.id.numpy()
        node = np.arange(graph.x.shape[0])
        dict_ = dict(zip(node, id_))
    else:
        dict_ = None
    return dict_


def create_graph(graph: torch_geometric.data.data.Data):
    g = to_networkx(graph)
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g,
        pos,
        node_text=get_node_txt(graph),
        node_text_position="top left",
        node_size=20,
    )
    fig = vis.create_figure()
    return fig
