import json
import torch

from eap.graph import Graph


def load_graph_from_json(json_path: str):
    """
    Load a Graph object from a JSON file.
    The JSON should have the following keys:
        1. 'cfg': Configuration dictionary, containing similar values to a TLens configuration object.
        2. 'nodes': Dict[str, bool] which maps a node name (i.e. 'm11' or 'a0.h11') to a boolean value, indicating if the node is part of the circuit.
        3. 'edges': Dict[str, Dict] which maps an edge name ('node->node') to a dictionary contains values 
        4. 'neurons': Optional[Dict[str, List[bool]]] which maps a node name (i.e. 'm11' or 'a0.h11') to a list of boolean values, indicating which of its neurons are part of the circuit.

    NOTE: This method isn't disk-space efficient, and shouldn't be used when the circuits contains edges between neuron-resolution nodes.
    """
    with open(json_path, 'r') as f:
        d = json.load(f)
        assert all([k in d.keys() for k in ['cfg', 'nodes', 'edges']]), "Bad input JSON format - Missing keys"

    g = Graph.from_model(d['cfg'], neuron_level=True, node_scores=True)
    any_node_scores, any_neurons, any_neurons_scores = False, False, False
    for name, node_dict in d['nodes'].items():
        if name == 'logits':
            continue
        g.nodes[name].in_graph = node_dict['in_graph']
        if 'score' in node_dict:
            any_node_scores = True
            g.nodes[name].score = node_dict['score']
        if 'neurons' in node_dict:
            any_neurons = True
            g.neurons_in_graph[g.forward_index(g.nodes[name])] = torch.tensor(node_dict['neurons']).float()
        if 'neurons_scores' in node_dict:
            any_neurons_scores = True
            g.neurons_scores[g.forward_index(g.nodes[name])] = torch.tensor(node_dict['neurons_scores']).float()
            
    if not any_node_scores:
        g.nodes_scores = None
    if not any_neurons:
        g.neurons_in_graph = None
    if not any_neurons_scores:
        g.neurons_scores = None        
    
    for name, info in d['edges'].items():
        g.edges[name].score = info['score']
        g.edges[name].in_graph = info['in_graph']
        
    return g

# TODO: MAYBE CHANGE THE WAY THAT THE EDGES ARE STORED IN THE GRAPH OBJECT TO THE TENSOR BASED ONE, TO SUPPORT EFFICIENT MASKING
def load_graph_from_pt(pt_path):
    """
    Load a graph object from a pytorch-serialized file.
    The file should contain a dict with the following items -
        1. 'cfg': Configuration dictionary, containing similar values to a TLens configuration object.
        2. 'src_nodes': Dict[str, bool] which maps a node name (i.e. 'm11' or 'a0.h11') to a boolean value, indicating if the node is part of the circuit.
        3. 'dst_nodes': List[str] containing the names of the possible destination nodes, in the same order as the edges tensor.
        4. 'edges': torch.tensor[n_src_nodes, n_dst_nodes], where each value in (src, dst) represents the edge score between the src node and dst node.
        5. 'edges_in_graph': torch.tensor[n_src_nodes, n_dst_nodes], where each value in (src, dst) represents if the edge is in the graph or not.
        6. 'neurons': [Optional] torch.tensor[n_src_nodes, d_model], where each value in (src, neuron) indicates whether the neuron is in the graph or not
    """
    d = torch.load(pt_path)
    required_keys = ['cfg', 'src_nodes', 'dst_nodes', 'edges_scores', 'edges_in_graph', 'nodes_in_graph']
    assert all([k in d.keys() for k in required_keys]), f"Bad torch circuit file format. Found keys - {d.keys()}, missing keys - {set(required_keys) - set(d.keys())}"
    assert d['edges_scores'].shape == d['edges_in_graph'].shape, "Bad edges array shape"
    # assert d['edges'].shape == d['edges_in_graph'].shape == (len(d['src_nodes']), len(d['dst_nodes'])), "Bad edges array shape"

    g = Graph.from_model(d['cfg'])

    g.in_graph[:] = d['edges_in_graph']
    g.scores[:] = d['edges_scores']
    g.nodes_in_graph[:] = d['nodes_in_graph']
    
    if 'nodes_scores' in d:
        g.nodes_scores = d['nodes_scores']
                
    if 'neurons_in_graph' in d:
        g.neurons_in_graph = d['neurons_in_graph']
    
    if 'neurons_scores' in d:
        g.neurons_scores = d['neurons_scores']

    return g