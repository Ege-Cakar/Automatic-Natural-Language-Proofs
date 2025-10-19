import networkx as nx
import pandas as pd
import json
from typing import Union, Dict, Optional
from collections import defaultdict
import os

def convert_to_networkx(
    dag_input: Union[str, Dict],
    save_dir: Optional[str] = None
) -> Union[nx.DiGraph, Dict]:
    """
    Convert a single DAG (as a JSON string or dict) or a dictionary of DAGs
    into NetworkX DiGraph(s). Optionally save the resulting graph(s) to disk
    in GraphML format (with primitive node attributes).

    Args:
        dag_input: A JSON string, a dict representing one DAG, or a dict of such.
        save_dir: If provided, path to a directory where the graph(s) will be
                  saved in GraphML format. Directory will be created if needed.

    Returns:
        A NetworkX DiGraph if given a single DAG, or a dict mapping keys to DiGraphs.
    """
    def _single_to_graph(dag: Dict) -> nx.DiGraph:
        G = nx.DiGraph()
        for idx, node in enumerate(dag["tactics"]):
            pos_line = node["pos"]["line"]
            pos_col  = node["pos"]["column"]
            end_line = node["endPos"]["line"]
            end_col  = node["endPos"]["column"]

            G.add_node(
                idx,
                tactic     = node["tactic"],
                proofState = node["proofState"],
                pos_line   = pos_line,
                pos_col    = pos_col,
                end_line   = end_line,
                end_col    = end_col,
                goals      = node["goals"],
            )
        for i in range(len(dag["tactics"]) - 1):
            G.add_edge(i, i + 1, type="temporal")
        # State-flow edges
        state_to_nodes = defaultdict(list)
        for idx, node in enumerate(dag["tactics"]):
            state_to_nodes[node["proofState"]].append(idx)
        for nodes in state_to_nodes.values():
            sorted_nodes = sorted(nodes)
            for u, v in zip(sorted_nodes, sorted_nodes[1:]):
                G.add_edge(u, v, type="state_flow")
        return G

    if isinstance(dag_input, str):
        dag_dict = json.loads(dag_input)
        result = _single_to_graph(dag_dict)
    elif isinstance(dag_input, dict) and "tactics" in dag_input:
        result = _single_to_graph(dag_input)
    elif isinstance(dag_input, dict):
        result = {key: convert_to_networkx(value) for key, value in dag_input.items()}
    else:
        raise ValueError("Input must be a JSON string, a DAG dict, or dict of DAGs")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(result, nx.DiGraph):
            path = os.path.join(save_dir, "dag.graphml")
            nx.write_graphml(result, path)
        else:
            for key, G in result.items():
                filename = f"{key}.graphml"
                path = os.path.join(save_dir, filename)
                nx.write_graphml(G, path)

    return result


def get_dags(csv_path: str) -> dict:
    """
    Read a CSV file containing DAGs and return a dictionary mapping Herald IDs to their DAGs

    Args:
        csv_path (str): Path to the CSV file containing the DAG data.

    Returns:
        dict: A dictionary mapping Herald IDs to their corresponding DAGs.
    """

    df = pd.read_csv(csv_path)
    proofs = {}

    for _, row in df.iterrows():
        if row["REPL Output"] == "Skipped as LEAN compilation failed":
            continue
        herald_id = row["Herald ID"]
        repl_json = row["REPL Output"]
        dag = json.loads(repl_json)
        proofs[herald_id] = dag

    return proofs