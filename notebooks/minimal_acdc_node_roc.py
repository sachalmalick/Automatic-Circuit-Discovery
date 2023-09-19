#%%

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")
import os
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px

#%%

# Set your root directory
ROOT_DIR = Path("/home/arthur/Documents/Automatic-Circuit-Discovery")
assert ROOT_DIR.exists(), f"I don't think your ROOT_DIR is correct (ROOT_DIR = {ROOT_DIR})"

# %%

TASK = "ioi"
METRIC = "kl_div"
FNAME = f"experiments/results/plots_data/acdc-{TASK}-{METRIC}-False-0.json"
FPATH = ROOT_DIR / FNAME
assert FPATH.exists(), f"I don't think your FNAME is correct (FPATH = {FPATH})"

# %%

data = json.load(open(FPATH, "r")) 

# %%

relevant_data = data["trained"]["random_ablation"]["ioi"]["kl_div"]["ACDC"]

# %%

node_tpr = relevant_data["node_tpr"]
node_fpr = relevant_data["node_fpr"]

# %%

# We would just plot these, but sometimes points are not on the Pareto frontier

def pareto_optimal_sublist(xs, ys):
    retx, rety = [], []
    for x, y in zip(xs, ys):
        for x1, y1 in zip(xs, ys):
            if x1 > x and y1 < y:
                break
        else:
            retx.append(x)
            rety.append(y)
    indices = sorted(range(len(retx)), key=lambda i: retx[i])
    return [retx[i] for i in indices], [rety[i] for i in indices]

# %%

pareto_node_tpr, pareto_node_fpr = pareto_optimal_sublist(node_tpr, node_fpr)

# %%

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(pareto_node_fpr),
        y=list(pareto_node_tpr),
        mode="lines",
        line=dict(shape="hv"),
        showlegend=False,
    ),
)

fig.update_layout(
    title="ROC Curve of number of Nodes recovered by ACDC",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
)

fig.show()

# %%