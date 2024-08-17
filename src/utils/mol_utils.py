import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch_geometric.data import Data
from ogb.utils import features


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
    pos = pos.numpy()
    rot = get_3d_rotation_mat_v3(pos)
    pos = np.matmul(rot, pos.T).T  # [3,3] \dot [3,N] -> [3,N] -> [N,3]
    return torch.tensor(pos, dtype=torch.float32)
