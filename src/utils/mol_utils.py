import numpy as np
from typing import Dict, Tuple
from scipy.spatial.transform import Rotation
import torch
from torch_geometric.data import Data
from ogb.utils import features


# position percentiles: stats from PCQM4M-v2 dataset after applying `rotate_3d_v3`
# percentiles: [0, 0.1, 0.5, 1, 5, 95, 99, 99.5, 99.9, 100]
# array([-14.25061607,  -8.12349338,  -6.46101438,  -5.63378452, -3.48104267,
#          3.0472374 ,   4.94877178,   5.70866934,  7.19100503,  13.97819328]) -> x-axis
# array([-14.17929077,  -6.57195779,  -4.89564619,  -4.11754894, -1.90409404,
#          6.28023171,   8.23885832,   8.89545588,  10.24105761,  16.76568604]) -> y-axis
# array([-12.95975018,  -5.30149424,  -3.8687721 ,  -2.98718095,  -0.90624478,
#          7.64074194,   9.72860246,  10.45567324,  11.93412937,  51.73352051]) -> z-axis
# a). range_min: 0.1 percentile, range_max: 99.9 percentile
RANGE_min_p1p = torch.tensor([-8.12, -6.57, -5.3]).float().view((1, -1))
RANGE_max_p1p = torch.tensor([7.19, 10.24, 11.93]).float().view((1, -1))
# b). range_min: 1 percentile, range_max: 99 percentile
RANGE_min_1p = torch.tensor([-5.63, -4.12, -2.99]).float().view((1, -1))
RANGE_max_1p = torch.tensor([4.95, 8.24, 9.73]).float().view((1, -1))

DICT_range = {"p1p": (RANGE_min_p1p, RANGE_max_p1p), "1p": (RANGE_min_1p, RANGE_max_1p)}


def _get_all_possible_attr(ls_feat_dims):
    max_dim = max(ls_feat_dims)
    ls_all = []
    for cur_dim in ls_feat_dims:
        div = max_dim // cur_dim
        res = max_dim - div * cur_dim
        cur_ls = list(range(cur_dim)) * div + list(range(res))
        ls_all.append(cur_ls)
    attr = torch.tensor(ls_all, dtype=torch.int64).T.clone()
    return attr


def read_complete_mol_features_ds():
    x = _get_all_possible_attr(features.get_atom_feature_dims())
    edge_attr = _get_all_possible_attr(features.get_bond_feature_dims())
    data = Data(edge_attr=edge_attr, x=x)
    print(f"Molecule dataset contains all possible features:\n{data}")
    return [data]


def read_complete_onedevice_features_ds():
    x = _get_all_possible_attr([10])
    edge_attr = _get_all_possible_attr([10, 10, 10, 91, 91])
    data = Data(edge_attr=edge_attr, x=x)
    print(f"OneDevice dataset contains all possible features:\n{data}")
    return [data]


def get_3d_rotation_mat():
    # https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations
    angles = (np.random.random(3) * 2 * np.pi).astype(np.float32)
    cos_alpha, cos_beta, cos_gamma = np.cos(angles)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles)

    yaw = np.array(
        [[cos_alpha, -sin_alpha, 0], [sin_alpha, cos_alpha, 0], [0, 0, 1]],
        dtype=np.float32,
    )

    pitch = np.array(
        [[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]], dtype=np.float32
    )

    roll = np.array(
        [[1, 0, 0], [0, cos_gamma, -sin_gamma], [0, sin_gamma, cos_gamma]],
        dtype=np.float32,
    )

    rot = np.matmul(np.matmul(yaw, pitch), roll)
    return rot


def rotate_3d(pos: torch.Tensor):
    # pos -> dim [N, 3]
    rot_tensor = torch.tensor(get_3d_rotation_mat())
    pos = torch.matmul(rot_tensor, pos.T).T  # [3,3] \dot [3,N] -> [3,N] -> [N,3]
    return pos


def rotate_3d_v2(pos: torch.Tensor):
    # pos -> dim [N, 3]
    r = Rotation.random()
    pos = r.apply(pos.numpy())
    pos = torch.from_numpy(pos).float()
    return pos


def get_3d_rotation_mat_v3(pos: np.ndarray):
    # pos -> dim [N, 3]
    # pos[0, :] -> (0,0,0)
    # https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations
    eps = 1e-12
    b = pos[1, :]
    # rotate 1. around x-axis, bring node-1 to x-z plane, i.e., y=0
    # below cos/sin settings ensure node-1's z-axis val is positive
    norm = np.max([np.sqrt(b[1] ** 2 + b[2] ** 2), eps])
    cos_gamma = b[2] / norm
    sin_gamma = b[1] / norm

    roll = np.array([[1, 0, 0], [0, cos_gamma, -sin_gamma], [0, sin_gamma, cos_gamma]])

    # rotate 2. around y-axis, bring node-1 to z-axis, i.e., x=0, y=0
    # below cos/sin settings ensure node-1's z-axis val is positive
    norm = np.max([np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2), eps])
    cos_beta = np.sqrt(b[1] ** 2 + b[2] ** 2) / norm
    sin_beta = -b[0] / norm

    pitch = np.array([[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]])

    rot = np.matmul(pitch, roll)

    if pos.shape[0] > 2:
        c = pos[2, :]
        c = np.matmul(rot, c)
        # rotate 3. around z-axis, bring node-2 to y-z plane, i.e., x=0
        norm = np.max([np.sqrt(c[0] ** 2 + c[1] ** 2), eps])
        cos_alpha = c[1] / norm
        sin_alpha = c[0] / norm

        yaw = np.array(
            [[cos_alpha, -sin_alpha, 0], [sin_alpha, cos_alpha, 0], [0, 0, 1]]
        )
        rot = np.matmul(yaw, rot)
    return rot


def rotate_3d_v3(pos: torch.Tensor):
    # pos -> dim [N, 3]
    # 0. move node-0 to the origin
    pos = pos - pos[0, :]
    if pos.shape[0] == 1:
        return pos
    pos = pos.numpy()
    rot = get_3d_rotation_mat_v3(pos)
    pos = np.matmul(rot, pos.T).T  # [3,3] \dot [3,N] -> [3,N] -> [N,3]
    return torch.tensor(pos, dtype=torch.float32)


def trans_rotate_3d_random(pos: torch.Tensor):
    pos = pos - pos.mean(dim=0, keepdim=True)
    return rotate_3d_v2(pos)


def discrete_pos(
    pos,
    num_bins,
    *,
    range_min: torch.Tensor = torch.tensor([[-6]], dtype=torch.float32),
    range_max: torch.Tensor = torch.tensor([[9]], dtype=torch.float32),
    **kwargs,
):
    # refer to: https://github.com/shamim-hussain/tgt/blob/697657b9806ba4c52642c80d4674291f41be5bc5/lib/training_schemes/pcqm/commons.py#L19
    range_min = range_min.float().view((1, -1))
    range_max = range_max.float().view((1, -1))
    range_bins = range_max - range_min
    pos = (pos - range_min) * ((num_bins - 1) / range_bins)
    pos = pos.long().clamp(0, num_bins - 1)
    return pos


def discrete_pos_v2(pos, num_bins, *, dict_bounds, **kwargs):
    # bins boundary is created and saved using percentile
    # AND with pos values inbetween [-1e-4,1e-4] removed when creating percentile
    # BECAUSE they are too many!
    # print("[DEBUG] invoking discrete_dist_v2")
    pos_clipped = torch.clamp(pos, min=-99, max=99).float()
    boundaries = dict_bounds[num_bins].float().to(pos_clipped.device)
    pos_bins = torch.bucketize(pos_clipped, boundaries) - 1
    # print(f"[DEBUG] pos_clipped:\n{pos_clipped}")
    # print(f"[DEBUG] pos_bins:\n{pos_bins}\n{pos_bins.shape}")
    # uniques, counts = torch.unique(pos_bins, return_counts=True)
    # print(f"[DEBUG] uniques:\n{uniques}\ncounts:\n{counts}\ncounts sum:\n{counts.sum()}")
    return pos_bins


def _trans_rot_pos(
    graph,
    node_structure_mapping: Dict[int, Tuple[str]],
):
    # 1. get the coord of nodes in the order of euler sequence
    idx_euler_order = [
        old_idx
        for old_idx, new_idx in sorted(
            node_structure_mapping.items(), key=lambda x: eval("+".join(x[1]))
        )
    ]
    if (
        hasattr(graph, "pos")
        and isinstance(graph.pos, torch.Tensor)
        and (not torch.isnan(graph.pos).any().item())
    ):
        pos = graph.pos[idx_euler_order]

        # 2. reset the coord by subtracting 1st node's coord in euler sequence
        pos = pos - pos[0:1]  # [N, 3]  translational invariant
    else:
        pos = torch.tensor([0, 0, 0], dtype=torch.float32)

    if torch.abs(pos).sum() > 1e-8:
        pos = rotate_3d_v3(pos)  # [N,3]  rotational invariant
    return pos, idx_euler_order


def decorate_molecules_with_3d_positions(
    graph,
    gtokenizer,
    node_structure_mapping: Dict[int, Tuple[str]],
    trim_zeros: bool = True,
):
    # 1. define two special tokens: icl_token, sep_token
    eos_token = gtokenizer.get_eos_token()
    icl_token = gtokenizer.get_icl_token()
    sep_token = gtokenizer.get_sep_token()

    # 2&3. translational & rotational invariant transformation of 3d-pos
    pos, idx_euler_order = _trans_rot_pos(graph, node_structure_mapping)

    # 4. obtain the tokens if the graph has 3d coordinate
    def _process_each_col(col_val: str):
        ls_col_val = list(col_val)
        # ls_col_val = _remove_lead_zero(ls_col_val)
        return [f"<{x}>" for x in ls_col_val]

    if torch.abs(pos).sum() > 1e-8:
        decimals = gtokenizer.config["semantics"].get("3d_decimals", 2)
        ls_coords = []
        pos_str = pos.round(decimals=decimals).numpy().astype(str)
        for idx, node_pos in enumerate(
            pos_str[1:], 1
        ):  # 1st node is translated to (0,0,0), so ignore it
            coords = [[sep_token] + _process_each_col(col_val) for col_val in node_pos]
            # 2nd node is rotated to (0,0,z), so ignore x,y coords
            # 3rd node is rotated to (0,y,z), so ignore x coords
            trim = max(0, 3 - idx) if trim_zeros else 0
            coords = coords[trim:]
            coords_flat = [token for each_coord in coords for token in each_coord]
            coords_flat = coords_flat[1:]  # remove the leading sep_token

            old_idx = idx_euler_order[idx]
            new_idx = node_structure_mapping[old_idx]

            node_with_coords = list(new_idx) + coords_flat
            ls_coords.append(node_with_coords)
            # print(node_with_coords)
        ls_coords = [token for each in ls_coords for token in each]
        ls_coords = [eos_token, icl_token] + ls_coords
    else:
        ls_coords = []
    return ls_coords
