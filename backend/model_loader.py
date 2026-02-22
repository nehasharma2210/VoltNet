# backend/model_loader.py
import os
import pickle
import numpy as np
import torch
import scipy.io as sio
import traceback
import re

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class SurrogateGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=128, n_layers=3, dropout=0.1, out_node_dim=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        for i in range(n_layers):
            ind = in_dim if i == 0 else hidden_dim
            conv = TransformerConv(ind, hidden_dim, edge_dim=edge_dim, heads=1, dropout=dropout)
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.res_projs.append(nn.Linear(ind, hidden_dim) if ind != hidden_dim else nn.Identity())
        self.node_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_node_dim))

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv, norm, proj in zip(self.layers, self.norms, self.res_projs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = norm(h)
            h = h + proj(h_in)
        return self.node_head(h)

def load_artifacts(artifact_dir=None, device='cpu', verbose=True):
    """
    Loads model, scalers, graph artifacts and baseline input for inference.
    If baseline_X is missing or too small, generate fallback baseline using scaler_x (inverse of zeros).
    Returns a dict of artifacts and dims.
    """
    try:
        if artifact_dir is None:
            artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        if verbose:
            print("Loading artifacts from:", artifact_dir)

        # 1) scalers.pkl
        scalers_fp = os.path.join(artifact_dir, "scalers.pkl")
        if not os.path.exists(scalers_fp):
            raise FileNotFoundError(f"scalers.pkl not found at {scalers_fp}")
        with open(scalers_fp, "rb") as f:
            meta = pickle.load(f)

        required_meta = ("scaler_x", "scaler_y", "scaler_e", "num_nodes", "in_dim", "out_dim")
        missing = [k for k in required_meta if k not in meta]
        if missing:
            raise KeyError(f"scalers.pkl is missing keys: {missing}. Available keys: {list(meta.keys())}")

        scaler_x = meta["scaler_x"]
        scaler_y = meta["scaler_y"]
        scaler_e = meta["scaler_e"]
        num_nodes = int(meta["num_nodes"])
        in_dim = int(meta["in_dim"])
        meta_out_dim = int(meta["out_dim"])
        edge_dim_from_meta = int(meta["edge_dim"]) if "edge_dim" in meta else None

        # 2) baseline_X (prefer .npy; fallback to .mat)
        baseline_fp = os.path.join(artifact_dir, "baseline_X.npy")
        baseline = None
        if os.path.exists(baseline_fp):
            baseline = np.load(baseline_fp)
        else:
            baseline_mat = os.path.join(artifact_dir, "baseline_X.mat")
            if os.path.exists(baseline_mat):
                d = sio.loadmat(baseline_mat)
                candidates = [(k, v) for k, v in d.items() if not k.startswith("__")]
                if candidates:
                    key, arr = max(candidates, key=lambda kv: np.array(kv[1]).size)
                    baseline = np.array(arr)

        # We'll validate and possibly replace baseline later after we know expected size.

        # 3) edge_index / edge_attr (.npy preferred)
        ei_fp = os.path.join(artifact_dir, "edge_index.npy")
        ea_fp = os.path.join(artifact_dir, "edge_attr.npy")
        edge_index = None
        edge_attr = None
        if os.path.exists(ei_fp) and os.path.exists(ea_fp):
            edge_index = np.load(ei_fp)
            edge_attr = np.load(ea_fp)
        else:
            ei_mat = os.path.join(artifact_dir, "edge_index.mat")
            ea_mat = os.path.join(artifact_dir, "edge_attr.mat")
            if os.path.exists(ei_mat) and os.path.exists(ea_mat):
                edge_index = sio.loadmat(ei_mat).get("edge_index")
                edge_attr = sio.loadmat(ea_mat).get("edge_attr")
        if edge_index is None or edge_attr is None:
            raise FileNotFoundError("edge_index/edge_attr not found in artifacts (.npy or .mat). Please save them from training.")

        ei = np.array(edge_index)
        if ei.ndim == 2 and ei.shape[0] == 2:
            pass
        elif ei.ndim == 2 and ei.shape[1] == 2:
            ei = ei.T
        else:
            raise ValueError(f"Unsupported edge_index shape: {ei.shape}")
        ei = ei.astype(np.int64)
        if ei.min() >= 1 and ei.max() <= num_nodes:
            if verbose:
                print("Detected 1-based edge_index; converting to 0-based.")
            ei = ei - 1
        if ei.min() < 0 or ei.max() >= num_nodes:
            raise IndexError(f"edge_index contains indices outside [0, {num_nodes-1}]. min={ei.min()} max={ei.max()}")

        ea = np.array(edge_attr)
        if ea.ndim == 1:
            ea = ea.reshape(-1, 1)
        if ea.shape[0] != ei.shape[1]:
            if ea.shape[0] == ei.shape[0] and ea.shape[1] == ei.shape[1]:
                ea = ea.T
            else:
                raise ValueError(f"edge_attr length {ea.shape[0]} doesn't match number of edges {ei.shape[1]}")
        edge_dim = ea.shape[1]

        # 4) checkpoint inspection to infer output dim
        model_path = os.path.join(artifact_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"best_model.pth not found in {artifact_dir}")

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        # normalize to a state_dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # find node_head.*.weight keys; choose the highest index (final linear)
        node_head_weights = []
        for k in state_dict.keys():
            m = re.match(r"node_head\.(\d+)\.weight$", k)
            if m:
                idx = int(m.group(1))
                node_head_weights.append((idx, k, state_dict[k]))
        out_dim_from_ckpt = None
        if node_head_weights:
            node_head_weights.sort(key=lambda t: t[0])
            final_idx, final_key, final_w = node_head_weights[-1]
            out_dim_from_ckpt = int(final_w.shape[0])
            if verbose:
                print("Found node_head weight key:", final_key, "-> inferred out_dim =", out_dim_from_ckpt)
        elif "node_head.2.weight" in state_dict:
            out_dim_from_ckpt = int(state_dict["node_head.2.weight"].shape[0])
            if verbose:
                print("Found node_head.2.weight -> out_dim =", out_dim_from_ckpt)
        else:
            # last fallback
            out_dim_from_ckpt = int(meta["out_dim"])
            if verbose:
                print("Could not find node_head weight keys; falling back to scalers.pkl['out_dim'] =", out_dim_from_ckpt)

        if out_dim_from_ckpt != meta_out_dim:
            print("WARNING: checkpoint out_dim != scalers.pkl['out_dim']. Checkpoint:", out_dim_from_ckpt, "scalers.pkl:", meta_out_dim)

        # Build model with inferred out_dim and load state
        model = SurrogateGNN(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=128, n_layers=3, dropout=0.1, out_node_dim=out_dim_from_ckpt)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print("Error loading state_dict into model (strict=False):", str(e))
            raise

        model.to(device).eval()

        # 5) Ensure baseline exists and has expected size; otherwise create fallback baseline
        expected_size = num_nodes * in_dim
        if baseline is None:
            print("baseline_X not found in artifacts -> creating fallback baseline from scaler_x (inverse of zeros).")
            # scaler_x expects shape (n_samples, expected_size)
            zeros = np.zeros((1, expected_size))
            try:
                fallback_flat = scaler_x.inverse_transform(zeros)  # yields feature_min (or scaled zero baseline)
                baseline = fallback_flat.reshape(num_nodes, in_dim)
            except Exception:
                # last resort: zeros
                baseline = np.zeros((num_nodes, in_dim), dtype=float)
        else:
            baseline = np.array(baseline)
            # try to reshape or fix
            if baseline.size == expected_size:
                baseline = baseline.reshape(num_nodes, in_dim)
            elif baseline.ndim == 2 and baseline.shape == (num_nodes, in_dim):
                pass
            else:
                # If smaller than expected, generate fallback and warn
                if baseline.size < expected_size:
                    print(f"WARNING: Provided baseline_X size {baseline.size} < expected {expected_size}. Generating fallback baseline from scaler_x.")
                    zeros = np.zeros((1, expected_size))
                    try:
                        fallback_flat = scaler_x.inverse_transform(zeros)
                        baseline = fallback_flat.reshape(num_nodes, in_dim)
                    except Exception:
                        baseline = np.zeros((num_nodes, in_dim), dtype=float)
                else:
                    baseline = baseline.flatten()[:expected_size].reshape(num_nodes, in_dim)

        if verbose:
            print(f"Loaded artifacts: num_nodes={num_nodes}, in_dim={in_dim}, out_dim(checkpoint)={out_dim_from_ckpt}, edge_dim={edge_dim}, edges={ei.shape[1]}")

        return {
            "model": model,
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "scaler_e": scaler_e,
            "edge_index": ei,
            "edge_attr": ea.astype(np.float32),
            "baseline_X": baseline,
            "num_nodes": num_nodes,
            "in_dim": in_dim,
            "out_dim": out_dim_from_ckpt,
            "meta_out_dim": meta_out_dim,
            "edge_dim": edge_dim,
            "scalers_meta": meta
        }

    except Exception as exc:
        print("Error in load_artifacts():", str(exc))
        traceback.print_exc()
        raise
