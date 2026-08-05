import re

import numpy as np
import torch
from torch_geometric.data import Data

EDGE_DIM = 9
NODE_FEAT_DIM = 5  # bias + one-hot layer type (input, conv, fc1, fc2)

# raw weight/bias magnitudes are tiny (typically 0.02-0.25 std), which starves
# the network of gradient signal; rescale into a more typical operating range.
FEATURE_SCALE = 10.0

LAYER_INPUT, LAYER_CONV, LAYER_FC1, LAYER_FC2 = range(4)


def _get_conv_layers(state_dict: dict):
    indices = sorted(
        int(m.group(1))
        for k in state_dict
        if (m := re.match(r"conv\.(\d+)\.weight", k))
    )
    return [(state_dict[f"conv.{i}.weight"].numpy(), state_dict[f"conv.{i}.bias"].numpy()) for i in indices]


def _pad_edge_feats(feats: np.ndarray) -> np.ndarray:
    n, d = feats.shape
    if d >= EDGE_DIM:
        return feats[:, :EDGE_DIM].astype(np.float32)
    padded = np.zeros((n, EDGE_DIM), dtype=np.float32)
    padded[:, :d] = feats
    return padded


def _node_feature(bias: float, layer_type: int) -> list:
    onehot = [0.0, 0.0, 0.0, 0.0]
    onehot[layer_type] = 1.0
    return [float(bias) * FEATURE_SCALE] + onehot


def build_graph(state_dict: dict) -> Data:
    conv_layers = _get_conv_layers(state_dict)
    fc1_w = state_dict["fc1.weight"].numpy()
    fc1_b = state_dict["fc1.bias"].numpy()
    fc2_w = state_dict["fc2.weight"].numpy()
    fc2_b = state_dict["fc2.bias"].numpy()

    node_features = [_node_feature(0.0, LAYER_INPUT)]
    layer_offsets = {"input": 0}
    offset = 1
    for li, (_, b) in enumerate(conv_layers):
        layer_offsets[f"conv{li}"] = offset
        node_features.extend(_node_feature(bi, LAYER_CONV) for bi in b)
        offset += len(b)
    layer_offsets["fc1"] = offset
    node_features.extend(_node_feature(bi, LAYER_FC1) for bi in fc1_b)
    offset += len(fc1_b)
    layer_offsets["fc2"] = offset
    node_features.extend(_node_feature(bi, LAYER_FC2) for bi in fc2_b)

    edge_srcs, edge_dsts, edge_feats = [], [], []

    def add_bipartite_edges(src_offset, n_src, dst_offset, n_dst, feats):
        src_idx = np.repeat(np.arange(n_src), n_dst) + src_offset
        dst_idx = np.tile(np.arange(n_dst), n_src) + dst_offset
        edge_srcs.append(src_idx)
        edge_dsts.append(dst_idx)
        edge_feats.append(_pad_edge_feats(feats))

    # input -> conv0: weight shape (out_ch, 1, k, k)
    w0, _ = conv_layers[0]
    out_ch0, in_ch0 = w0.shape[0], w0.shape[1]
    feats0 = w0.reshape(out_ch0 * in_ch0, -1)
    add_bipartite_edges(layer_offsets["input"], in_ch0, layer_offsets["conv0"], out_ch0, feats0)

    # conv[i-1] -> conv[i]: weight shape (out_ch, in_ch, k, k)
    for li in range(1, len(conv_layers)):
        w, _ = conv_layers[li]
        out_ch, in_ch = w.shape[0], w.shape[1]
        feats = w.reshape(out_ch * in_ch, -1)
        add_bipartite_edges(
            layer_offsets[f"conv{li - 1}"], in_ch, layer_offsets[f"conv{li}"], out_ch, feats
        )

    # last conv -> fc1: bridge each (channel, fc unit) pair with spatial-position summary stats
    last_li = len(conv_layers) - 1
    last_channels = conv_layers[-1][0].shape[0]
    fc_hidden = fc1_w.shape[0]
    spatial = int(round((fc1_w.shape[1] / last_channels) ** 0.5))
    fc1_w_reshaped = fc1_w.reshape(fc_hidden, last_channels, spatial * spatial)
    # -> (last_channels, fc_hidden, spatial*spatial), src=channel outer, dst=fc unit inner
    fc1_w_reshaped = fc1_w_reshaped.transpose(1, 0, 2)
    stats = np.stack(
        [
            fc1_w_reshaped.mean(axis=-1),
            fc1_w_reshaped.std(axis=-1),
            fc1_w_reshaped.min(axis=-1),
            fc1_w_reshaped.max(axis=-1),
            np.linalg.norm(fc1_w_reshaped, axis=-1),
        ],
        axis=-1,
    ).reshape(last_channels * fc_hidden, 5)
    add_bipartite_edges(
        layer_offsets[f"conv{last_li}"], last_channels, layer_offsets["fc1"], fc_hidden, stats
    )

    # fc1 -> fc2: weight shape (n_classes, fc_hidden)
    n_classes = fc2_w.shape[0]
    feats_fc2 = fc2_w.T.reshape(fc_hidden * n_classes, 1)
    add_bipartite_edges(layer_offsets["fc1"], fc_hidden, layer_offsets["fc2"], n_classes, feats_fc2)

    src = np.concatenate(edge_srcs)
    dst = np.concatenate(edge_dsts)
    feats = np.concatenate(edge_feats, axis=0) * FEATURE_SCALE

    # make undirected so information can flow both ways during message passing
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_attr = np.concatenate([feats, feats], axis=0)

    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)


def build_graph_from_file(weights_path: str) -> Data:
    state_dict = torch.load(weights_path, map_location="cpu")
    return build_graph(state_dict)
