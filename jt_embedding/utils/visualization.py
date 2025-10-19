import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Union
import os



def visualize_DAG(
    G_or_path: Union[str, nx.DiGraph],
    save_path: Optional[str] = None
    ) -> None:
    """
    Visualize a NetworkX DiGraph representing a DAG, or load one from a .graphml file.

    Args:
        G_or_path: Either a networkx.DiGraph object or a path to a .graphml file.
        save_path: If provided, save the rendered plot to this file (PNG). Otherwise show it.
    """
    if isinstance(G_or_path, str):
        if not os.path.exists(G_or_path):
            raise FileNotFoundError(f"No such file: {G_or_path}")
        G = nx.read_graphml(G_or_path)
        try:
            G = nx.relabel_nodes(G, lambda n: int(n))
        except ValueError:
            pass
    elif isinstance(G_or_path, nx.DiGraph):
        G = G_or_path
    else:
        raise ValueError("Input must be a networkx.DiGraph or a path to a .graphml file")

    pos = nx.spring_layout(G, seed=42)

    node_labels = {}
    for n, data in G.nodes(data=True):
        tactic = data.get('tactic', '')
        short = (tactic[:30] + '…') if len(tactic) > 30 else tactic
        state = data.get('proofState', '?')
        node_labels[n] = f"{n}\n{short}\nstate={state}"

    edge_labels = {(u, v): d.get('type', '') for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    temporal = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'temporal']
    state   = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'state_flow']
    nx.draw_networkx_edges(G, pos, edgelist=temporal, width=2, edge_color='tab:blue')
    nx.draw_networkx_edges(G, pos, edgelist=state,   width=2, edge_color='tab:orange', style='dashed')

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=7)

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()