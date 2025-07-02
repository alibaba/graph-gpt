# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import os.path as osp
import shutil
import tarfile
import time
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

import torch_geometric.typing
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import sort_edge_index, to_networkx
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import InMemoryDataset, Data
from torch_sparse.metis import weight2metis
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem

from multiprocessing import Pool

from . import spice2graph_full_utils as spice2graph

# import spice2graph_full_utils as spice2graph  # when running this script only inside this dir

CHIRAL_CENTERS = ["R", "S"]
NUM_CONFS = 40


def mol2coords(mol: Chem.Mol) -> np.ndarray:
    """
    Generate 3D RDKIT coordinates from the molecule.
    (Adapted from https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/pretrained_compound/ChemRL/GEM-2/pahelix/utils/compound_tools.py)

    Args:
        mol (Chem.Mol): The molecule object.

    Returns:
        np.ndarray: The 3D coordinates of the molecule.
    """
    dtype = np.float32
    num_threads = 1  # 0: use all threads
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(
            new_mol, numConfs=NUM_CONFS, numThreads=num_threads
        )
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol, numThreads=num_threads)
        new_mol = Chem.RemoveHs(new_mol)
        index, _ = min(enumerate(res), key=lambda x: x[1])
        conf = new_mol.GetConformer(id=index)
    except Exception:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        conf = new_mol.GetConformer()

    if new_mol.GetAtomWithIdx(0).GetAtomicNum() == 0:
        return np.zeros((new_mol.GetNumAtoms(), 3), dtype=dtype)
    coords = conf.GetPositions()
    coords = coords[: new_mol.GetNumAtoms()].astype(dtype)
    return coords


def chiral_centers_feature(x_num, mol):
    # https://github.com/graphcore/ogb-lsc-pcqm4mv2/blob/9c206603eab62f09d61e649a778ac8efe251dede/data_utils/pcq_dataset_28features.py#L235
    # Is chiral center (+1)
    features = np.zeros([x_num], dtype=np.int64)
    for idx, type_ in Chem.FindMolChiralCenters(mol):
        # 0 for not center
        features[idx] = CHIRAL_CENTERS.index(type_) + 1
    return features


def mol2graph_basic(mol):
    # copied from `ogb/utils/mol.py::smiles2graph`
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


def mol2graph_cc(mol):
    # compared to `mol2graph_basic`, replace `chiral-tag` with `chiral-center`
    graph = mol2graph_basic(mol)
    x = graph["node_feat"]
    features = chiral_centers_feature(x.shape[0], mol)
    x[:, 1] = features
    graph["node_feat"] = x.copy()
    return graph


def mol2graph_pos(mol):
    # refer to: https://github.com/lsj2408/Transformer-M/blob/main/Transformer-M/data/wrapper.py
    # add 3D position
    try:
        graph = mol2graph_basic(mol)
        # positions
        graph["position"] = mol.GetConformer().GetPositions()
        return graph
    except:
        return None


def mol2graph_pos_cc(mol):
    try:
        graph = mol2graph_cc(mol)
        # positions
        graph["position"] = mol.GetConformer().GetPositions()
        return graph
    except:
        return None


def mol2graph_pos_cc_rdkit_pos(mol):
    try:
        graph = mol2graph_cc(mol)
        # positions
        graph["position"] = mol.GetConformer().GetPositions()
        graph["pos_rdkit"] = mol2coords(mol)
        return graph
    except:
        return None


def smiles2graph_basic(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    return mol2graph_basic(mol)


def smiles2graph_cc(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    return mol2graph_cc(mol)


def smiles2graph_cc_rdkit_pos(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    graph = mol2graph_cc(mol)
    graph["pos_rdkit"] = mol2coords(mol)
    return graph


def smiles2graph_extra(smiles_string):
    """
    COPIED and modified from `ogb/utils/mol.py::smiles2graph`
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        # compared to original script, ONLY below 3 lines got modified
        z = atom.GetAtomicNum()
        ls_extra = [row(z), group(z), family(z)]
        atom_features_list.append(atom_to_feature_vector(atom) + ls_extra)
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


# below funcs `row`, `group`, `family` copied from https://github.com/graphcore/ogb-lsc-pcqm4mv2/blob/main/data_utils/feature_generation/generic_features.py
def row(z):
    _pt_row_sizes = (2, 8, 8, 18, 18, 32, 32)
    """
    Returns the periodic table row of the element.
    """
    total = 0
    if 57 <= z <= 71:
        return 8
    if 89 <= z <= 103:
        return 9
    for i, size in enumerate(_pt_row_sizes):
        total += size
        if total >= z:
            return i + 1
    return 8


def group(z):
    """
    Returns the periodic table group of the element.
    """
    if z == 1:
        return 1
    if z == 2:
        return 18
    if 3 <= z <= 18:
        if (z - 2) % 8 == 0:
            return 18
        if (z - 2) % 8 <= 2:
            return (z - 2) % 8
        return 10 + (z - 2) % 8

    if 19 <= z <= 54:
        if (z - 18) % 18 == 0:
            return 18
        return (z - 18) % 18

    if (z - 54) % 32 == 0:
        return 18
    if (z - 54) % 32 >= 18:
        return (z - 54) % 32 - 14
    if 57 <= z <= 71:
        return 3
    if 89 <= z <= 103:
        return 3
    return (z - 54) % 32


def family(z):
    """
    Returns the periodic table family of the element.
    0: Other non metals
    1: Alkali metals
    2: Alkaline earth metals
    3: Transition metals
    4: Lanthanides
    5: Actinides
    6: Other metals
    7: Metalloids
    8: Halogens
    9: Noble gases
    """
    if z in [1, 6, 7, 8, 15, 16, 34]:
        return 0
    if z in [3, 11, 19, 37, 55, 87]:
        return 1
    if z in [4, 12, 20, 38, 56, 88]:
        return 2
    if z in range(57, 72):
        return 4
    if z in range(89, 104):
        return 5
    if z in [13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 166]:
        return 6
    if z in [5, 14, 32, 33, 51, 52]:
        return 7
    if z in [9, 17, 35, 53, 85, 117]:
        return 8
    if z in [2, 10, 18, 36, 54, 86, 118]:
        return 9
    else:
        return 3


class PygPCQM4Mv2ExtraDataset(PygPCQM4Mv2Dataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=smiles2graph_extra,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, smiles2graph, transform, pre_transform)

    @property
    def processed_file_names(self):
        return "geometric_data_processed_extra.pt"


class PygPCQM4Mv2PosDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=None,
        sdf2graph=None,
        transform=None,
        pre_transform=None,
    ):
        """
        Pytorch Geometric PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m-v2
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        refer to: https://github.com/lsj2408/Transformer-M/blob/main/Transformer-M/data/wrapper.py
        """

        self.original_root = root
        if smiles2graph is None:
            self.smiles2graph = smiles2graph_basic
        else:
            self.smiles2graph = smiles2graph
        if sdf2graph is None:
            self.sdf2graph = mol2graph_pos
        else:
            self.sdf2graph = sdf2graph
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = (
            "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
        )
        self.pos_url = (
            "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz"
        )

        # check version and update if necessary
        if osp.isdir(self.folder) and (
            not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)

        fn_processed = self.processed_paths[0]
        print(f"Loading data from {fn_processed}")
        self.data, self.slices = torch.load(fn_processed)

    @property
    def raw_file_names(self):
        return ["data.csv.gz", "pcqm4m-v2-train.sdf"]

    @property
    def processed_file_names(self):
        return "geometric_data_processed_3dm_v2.pt"

    def download(self):
        if int(os.environ.get("RANK", 0)) == 0:
            # This is for multiple GPUs in one machine. The data is shared!
            if decide_download(self.url):
                path = download_url(self.url, self.original_root)
                extract_zip(path, self.original_root)
                # zip contains folder `pcqm4m-v2/raw`, so when extracting, the file will go to `self.raw_dir` folder
                os.unlink(path)
            else:
                print(f"Stop download {self.url}.")
                # exit(-1)

            if decide_download(self.pos_url):
                path = download_url(self.pos_url, self.raw_dir)
                tar = tarfile.open(path, "r:gz")
                filenames = tar.getnames()
                for file in filenames:
                    tar.extract(file, self.raw_dir)
                tar.close()
                os.unlink(path)
            else:
                print(f"Stop download {self.pos_url}.")
                # exit(-1)
        else:
            from torch_geometric.data.dataset import files_exist

            while not files_exist(self.raw_paths):
                print(f"sleep for RANK {os.environ.get('RANK', 0)} ...")
                time.sleep(3)

    def _preprocess(self):
        max_len, max_len2 = int(1e8), int(1e8)  # int(1e4)  For local testing
        processes = 20
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        graph_pos_list = Chem.SDMolSupplier(
            osp.join(self.raw_dir, "pcqm4m-v2-train.sdf"), removeHs=True
        )  # BE CAREFUL of `rdkit` version:: MAY cause incompatibility between `SMILES` and `SDF`
        # First 10k mol, 3.9% has different feat between `SMILES` and `SDF`
        homolumogap_list = data_df["homolumogap"]
        num_3d = len(graph_pos_list)
        num_all = len(homolumogap_list)
        print(f"Totally {num_all} molecules, with {num_3d} having 3D positions!")

        print(
            f"Extracting 3D positions of {num_3d} molecules from SDF files for Training Data..."
        )
        train_data_with_position_list = []
        with Pool(processes=processes) as pool:
            iter = pool.imap(self.sdf2graph, graph_pos_list)

            for i, graph in tqdm(enumerate(iter), total=num_3d):
                if i >= max_len:
                    print(f"break at {i}")
                    break
                try:
                    data = Data()
                    homolumogap = homolumogap_list[i]

                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]

                    data.__num_nodes__ = int(graph["num_nodes"])
                    data.edge_index = torch.from_numpy(graph["edge_index"]).to(
                        torch.int64
                    )
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(
                        torch.int64
                    )
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.from_numpy(graph["position"]).to(torch.float32)

                    train_data_with_position_list.append(data)
                except:
                    continue
        print(
            f"Done extracting 3D positions of {len(train_data_with_position_list)}/{num_3d}!"
        )

        smiles_list = data_df["smiles"][num_3d:].tolist()[:max_len2]
        homolumogap_list = homolumogap_list[num_3d:].tolist()[:max_len2]
        print(
            f"Converting SMILES strings of {len(smiles_list)} molecules into graphs..."
        )
        data_list = []
        with Pool(processes=processes) as pool:
            iter = pool.imap(self.smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]

                    data.__num_nodes__ = int(graph["num_nodes"])
                    data.edge_index = torch.from_numpy(graph["edge_index"]).to(
                        torch.int64
                    )
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(
                        torch.int64
                    )
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.zeros(data.__num_nodes__, 3).to(torch.float32)

                    data_list.append(data)
                except Exception as inst:
                    print(type(inst))
                    print(inst.args)
                    print(inst)
                    continue
        print(f"Done converting SMILES strings of {len(data_list)}/{len(smiles_list)}!")

        data_list = train_data_with_position_list + data_list

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["train"]])
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["valid"]])
        assert all([torch.isnan(data_list[i].y)[0] for i in split_dict["test-dev"]])
        assert all(
            [torch.isnan(data_list[i].y)[0] for i in split_dict["test-challenge"]]
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        return data_list

    def process(self):
        data_list = self._preprocess()
        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, "split_dict.pt"))
        )
        return split_dict


class PygPCQM4Mv2CCDataset(PygPCQM4Mv2Dataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=None,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, smiles2graph_cc, transform, pre_transform)

    @property
    def processed_file_names(self):
        return "geometric_data_processed_cc.pt"

    def process(self):
        max_len = 10000
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        smiles_list = data_df["smiles"][:max_len]
        homolumogap_list = data_df["homolumogap"]

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        # assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        # assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        # assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        # assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


class PygPCQM4Mv2PosCCDataset(PygPCQM4Mv2PosDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=None,
        sdf2graph=None,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(root, smiles2graph=smiles2graph_cc, sdf2graph=mol2graph_pos_cc)

    @property
    def processed_file_names(self):
        # `cc` means `Chiral-center` as the 2nd atomic feature
        return "geometric_data_processed_3d_cc.pt"


class PygPCQM4Mv2RdkitPosCCDataset(PygPCQM4Mv2PosDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=None,
        sdf2graph=None,
        transform=None,
        pre_transform=None,
    ):
        self.original_root = root
        self.download_rdkit_pos()
        super().__init__(root, smiles2graph=smiles2graph_cc, sdf2graph=mol2graph_pos_cc)

    def download_rdkit_pos(self):
        fn = os.path.join(
            self.original_root, "pcqm4m-v2", "processed", "rdkit_coords.parquet"
        )
        if int(os.environ.get("RANK", 0)) == 0:
            if not os.path.exists(fn):
                # This is for multiple GPUs in one machine. The data is shared!
                rdkit_url = "https://huggingface.co/datasets/shamim-hussain/pcqm/resolve/main/rdkit_coords.parquet"
                if decide_download(rdkit_url):
                    path = download_url(rdkit_url, self.processed_dir)
                    os.unlink(path)
                else:
                    print(f"Stop download {rdkit_url}.")
                    # exit(-1)
                # OR `wget -O rdkit_coords.parquet https://huggingface.co/datasets/shamim-hussain/pcqm/resolve/main/rdkit_coords.parquet?download=true`
                # refer to: https://github.com/shamim-hussain/tgt/blob/697657b9806ba4c52642c80d4674291f41be5bc5/download_data.sh#L15C1-L15C129
            else:
                print(f"rdkit pos already exists: {fn}")
        else:
            while not os.path.exists(fn):
                print(f"sleep for RANK {os.environ.get('RANK', 0)} ...")
                time.sleep(3)

    @property
    def processed_file_names(self):
        # `cc` means `Chiral-center` as the 2nd atomic feature
        # use rdkit generated 3D position as additional inputs
        return "geometric_data_processed_rdkit_3d_cc.pt"

    def _process_rdkit_pos(self, data_list):
        import pyarrow.dataset as pds

        max_len = int(1e8)
        coords_file = os.path.join(self.processed_dir, "rdkit_coords.parquet")
        coords_tab = pds.dataset(coords_file).to_table()
        ls_all_idx = []
        ls_inconsistent_idx = []
        for idx, pos in tqdm(zip(coords_tab["idx"], coords_tab["rdkit_coords"])):
            idx = int(idx.as_py())
            if idx >= max_len:
                print(f"break at {idx}")
                break
            pos = pos.values.to_numpy().reshape((-1, 3))
            nodes1 = data_list[idx].x.shape[0]
            nodes2 = pos.shape[0]
            if nodes1 != nodes2:
                print(f"idx: {idx}, {nodes1} != {nodes2}")
                ls_inconsistent_idx.append((idx, nodes1, nodes2))
                assert nodes1 < nodes2
            data_list[idx].rdkit_pos = torch.from_numpy(pos[:nodes1, :]).to(
                torch.float32
            )
            ls_all_idx.append(idx)
        print(
            f"Totally {len(ls_inconsistent_idx)} molecules have different number of atoms compare to our processed files"
        )

        # below mols in test-challenge, rdkit-pos not calculated yet
        remaining_mols = list(set(list(range(len(data_list)))) - set(ls_all_idx))
        for idx in tqdm(sorted(remaining_mols)):
            nodes1 = data_list[idx].x.shape[0]
            data_list[idx].rdkit_pos = torch.zeros(nodes1, 3).to(torch.float32)
        print(f"Set {len(remaining_mols)} mols' rdkit_pos to all 0's!")

    def process(self):
        data_list = self._preprocess()
        self._process_rdkit_pos(data_list)

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


def smiles2graph_with_try(smiles_string):
    try:
        graph = smiles2graph_basic(smiles_string)
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        graph = None
    return graph


class PygCEPDBDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=None,
        transform=None,
        pre_transform=None,
    ):
        """
        modified from ogb/lsc/pcqm4mv2_pyg.py::PygPCQM4Mv2Dataset
        Pytorch Geometric CEPDB dataset object
            - root (str): the dataset folder will be located at root/cepdb
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        """

        self.original_root = root
        self.smiles2graph = smiles2graph_with_try
        self.folder = osp.join(root, "cepdb")
        self.version = 1

        # md5sum:
        self.url = "https://figshare.com/ndownloader/files/17294444"

        # check version and update if necessary
        if osp.isdir(self.folder) and (
            not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("CEPDB dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super(PygCEPDBDataset, self).__init__(self.folder, transform, pre_transform)

        print(f"Loading data from {self.processed_paths[0]}")
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "17294444"

    @property
    def processed_file_names(self):
        # "geometric_data_processed.pt" -> y: homo-lumo "geometric_data_processed_full.pt" -> y: 7 properties
        return "geometric_data_processed_full.pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        processes = 10
        data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names))
        smiles_list = data_df["SMILES_str"]
        # vals = data_df[["e_gap_alpha"]].values.tolist()
        vals = data_df[
            ["mass", "pce", "voc", "jsc", "e_homo_alpha", "e_gap_alpha", "e_lumo_alpha"]
        ].values.tolist()

        print("Converting SMILES strings into graphs...")
        data_list = []
        with Pool(processes=processes) as pool:
            iter = pool.imap(self.smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(vals)):
                try:
                    data = Data()

                    val = vals[i]
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]

                    data.__num_nodes__ = int(graph["num_nodes"])
                    data.edge_index = torch.from_numpy(graph["edge_index"]).to(
                        torch.int64
                    )
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(
                        torch.int64
                    )
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([val])

                    data_list.append(data)
                except Exception as inst:
                    print(type(inst))
                    print(inst.args)
                    print(inst)
                    print(f"[Warning] Skip {smiles_list[i]}")
                    continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


class PygANI1Dataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=None,
        transform=None,
        pre_transform=None,
    ):
        """
        modified from ogb/lsc/pcqm4mv2_pyg.py::PygPCQM4Mv2Dataset
        Pytorch Geometric ANI-1 dataset object
            - root (str): the dataset folder will be located at root/cepdb
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        """

        self.original_root = root
        self.smiles2graph = smiles2graph_with_try
        self.folder = osp.join(root, "ani-1")
        self.version = 1

        # md5sum:
        self.url = "https://figshare.com/ndownloader/files/9057631"

        # check version and update if necessary
        if osp.isdir(self.folder) and (
            not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("ANI-1 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super(PygANI1Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "9057631"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        import pyanitools as pya

        hd5_files = [f"ani_gdb_s0{n}.h5" for n in range(1, 9)]
        processes = 10
        valid_smiles_list = []
        data_list = []
        for hdf5file in hd5_files:
            # Construct the data loader class
            adl = pya.anidataloader(osp.join(self.raw_dir, hdf5file))
            ls = [("".join(data["smiles"]), data["energies"][0]) for data in adl]
            # only select one conformation energy
            # Closes the H5 data file
            adl.cleanup()
            smiles_list = [x[0] for x in ls]
            energy_list = [x[1] for x in ls]

            print(f"Converting SMILES strings into graphs for {hdf5file}...")
            with Pool(processes=processes) as pool:
                iter = pool.imap(self.smiles2graph, smiles_list)

                for i, graph in tqdm(enumerate(iter), total=len(energy_list)):
                    try:
                        data = Data()

                        energy = energy_list[i]
                        assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                        assert len(graph["node_feat"]) == graph["num_nodes"]

                        data.__num_nodes__ = int(graph["num_nodes"])
                        data.edge_index = torch.from_numpy(graph["edge_index"]).to(
                            torch.int64
                        )
                        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(
                            torch.int64
                        )
                        data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                        data.y = torch.Tensor([energy])

                        data_list.append(data)
                        valid_smiles_list.append(smiles_list[i])
                    except Exception as inst:
                        print(type(inst))
                        print(inst.args)
                        print(inst)
                        print(f"[Warning] Skip {smiles_list[i]}")
                        continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

        df = pd.DataFrame(valid_smiles_list, columns=["smiles"])
        df.to_csv(osp.join(self.raw_dir, "smiles.csv"), index=False)


class PygZINCDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=None,
        transform=None,
        pre_transform=None,
        subset=11,
    ):
        """
        modified from ogb/lsc/pcqm4mv2_pyg.py::PygPCQM4Mv2Dataset
        Pytorch Geometric Zinc Clean Leads dataset object
            - root (str): the dataset folder will be located at root/cepdb
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
            - transform (callable):
            - pre_transform (callable):
            - subset (int): 11-> clean-leads, 4.6 M; 16-> all-clean, 16.4 M
                * 11 -> http://zinc12.docking.org/subsets/clean-leads
                * 16 -> http://zinc12.docking.org/subsets/all-clean
        """

        self.original_root = root
        self.smiles2graph = smiles2graph_with_try
        self.folder = osp.join(root, "zinc")
        self.version = 1
        self.subset = subset

        # let's only consider at reference PH=7 first, so use *_p0.smi.gz
        self.urls = [
            f"http://zinc12.docking.org/db/bysubset/{subset}/{subset}_prop.xls",
            # f"http://zinc12.docking.org/db/bysubset/{subset}/{subset}_p0.smi.gz",
            # f"http://zinc12.docking.org/db/bysubset/{subset}/{subset}_p1.smi.gz",
            # f"http://zinc12.docking.org/db/bysubset/{subset}/{subset}_p2.smi.gz",
            # f"http://zinc12.docking.org/db/bysubset/{subset}/{subset}_p3.smi.gz"
        ]

        # check version and update if necessary
        if osp.isdir(self.folder) and (
            not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print(f"Zinc {self.subset} dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super(PygZINCDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [url.split("/")[-1] for url in self.urls]

    @property
    def processed_file_names(self):
        # "geometric_data_processed_{self.subset}.pt" -> no y  "geometric_data_processed_{self.subset}_full.pt" -> y: 9 properties
        return f"geometric_data_processed_{self.subset}_full.pt"

    def download(self):
        for url in self.urls:
            if decide_download(url):
                path = download_url(url, self.original_root)
                if path.endswith("gz"):
                    extract_zip(path, self.original_root)
                os.unlink(path)
            else:
                print("Stop download.")
                exit(-1)

    def process(self):
        processes = 20
        data_list = []
        print(
            f"processing {len(self.raw_file_names)} raw files: {self.raw_file_names} ..."
        )
        for raw_file in self.raw_file_names:
            if raw_file.endswith("xls"):
                data_df = pd.read_csv(
                    osp.join(self.raw_dir, raw_file),
                    sep="\t",
                )
                # data_df = data_df.head(1000)  # for test only
                smiles_list = data_df["SMILES"]
            else:
                data_df = pd.read_csv(
                    osp.join(self.raw_dir, raw_file),
                    sep=" ",
                    header=None,
                    names=["smiles", "zinc_id"],
                )
                smiles_list = data_df["smiles"]
            vals = data_df[
                [
                    "MWT",
                    "LogP",
                    "Desolv_apolar",
                    "Desolv_polar",
                    "HBD",
                    "HBA",
                    "tPSA",
                    "Charge",
                    "NRB",
                ]
            ].values.tolist()

            print(f"Converting SMILES strings from {raw_file} into graphs ...")

            with Pool(processes=processes) as pool:
                iter = pool.imap(self.smiles2graph, smiles_list)

                for i, graph in tqdm(enumerate(iter), total=len(smiles_list)):
                    try:
                        data = Data()

                        val = vals[i]
                        assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                        assert len(graph["node_feat"]) == graph["num_nodes"]

                        data.__num_nodes__ = int(graph["num_nodes"])
                        data.edge_index = torch.from_numpy(graph["edge_index"]).to(
                            torch.int64
                        )
                        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(
                            torch.int64
                        )
                        data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                        data.y = torch.Tensor([val])

                        data_list.append(data)
                    except Exception as inst:
                        print(type(inst))
                        print(inst.args)
                        print(inst)
                        print(f"[Warning] Skip {smiles_list[i]}")
                        continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


class EdgeBalancedClusterData(ClusterData):
    def __init__(
        self,
        data,
        num_parts: int,
        recursive: bool = False,
        save_dir: Optional[str] = None,
        log: bool = True,
        balance_edge: bool = False,
    ):
        self.balance_edge = balance_edge
        super().__init__(
            data, num_parts=num_parts, recursive=recursive, save_dir=save_dir, log=log
        )

    def _metis(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        # Computes a node-level partition assignment vector via METIS.

        # Calculate CSR representation:
        row, col = sort_edge_index(edge_index, num_nodes=num_nodes)
        rowptr = index2ptr(row, size=num_nodes)

        # Compute METIS partitioning:
        cluster: Optional[Tensor] = None

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            try:
                if self.balance_edge:
                    node_weight = get_edge_balanced_node_weight(
                        rowptr.cpu(), col.cpu(), num_nodes
                    )
                    cluster = torch.ops.torch_sparse.partition2(
                        rowptr.cpu(),
                        col.cpu(),
                        None,
                        node_weight,
                        self.num_parts,
                        self.recursive,
                    ).to(edge_index.device)
                else:
                    cluster = torch.ops.torch_sparse.partition(
                        rowptr.cpu(),
                        col.cpu(),
                        None,
                        self.num_parts,
                        self.recursive,
                    ).to(edge_index.device)
            except (AttributeError, RuntimeError):
                pass

        if cluster is None and torch_geometric.typing.WITH_METIS:
            if self.balance_edge:
                raise NotImplementedError(
                    f"Edge-balanced Metis is not implemented in '{self.__class__.__name__}' with 'pyg-lib'"
                )
            cluster = pyg_lib.partition.metis(
                rowptr.cpu(),
                col.cpu(),
                self.num_parts,
                recursive=self.recursive,
            ).to(edge_index.device)

        if cluster is None:
            raise ImportError(
                f"'{self.__class__.__name__}' requires either "
                f"'pyg-lib' or 'torch-sparse'"
            )

        return cluster


def get_edge_balanced_node_weight(rowptr, col, num_nodes):
    # refer to: https://github.com/rusty1s/pytorch_sparse/blob/master/torch_sparse/metis.py
    #     func:: `partition`
    # 1. pre-calculate node_weight for edge balancing
    node_weight = torch.zeros(num_nodes, dtype=col.dtype)
    node_weight.scatter_add_(0, col, torch.ones_like(col))
    # 2. calculate node_weight for input to `torch.ops.torch_sparse.partition2`
    assert (
        node_weight.numel() == rowptr.numel() - 1
    ), f"{node_weight.numel()} != {rowptr.numel() - 1} !!!"
    node_weight = node_weight.view(-1).detach().cpu()
    if node_weight.is_floating_point():
        node_weight = weight2metis(node_weight)
    return node_weight


def obtain_graph_wgts(dataset: InMemoryDataset, idx: torch.Tensor):
    # weight graph by their number of eulerian paths, equivalently number of nodes
    if hasattr(dataset._data, "_num_nodes"):
        cnt_nodes = torch.tensor(dataset._data._num_nodes, dtype=torch.int64)
    elif hasattr(dataset, "x"):
        x_idx = dataset.slices["x"]
        # cnt of nodes of all graphs
        cnt_nodes = x_idx[1:] - x_idx[:-1]
    else:
        raise ValueError(
            "Cannot obtain graph weights because `dataset._data._num_nodes` or `dataset.x` NOT exists!"
        )
    # cnt of nodes of given graphs (by their idx)
    cnt_nodes_specific = cnt_nodes[idx].numpy()
    graph_wgts = cnt_nodes_specific / cnt_nodes_specific.sum()
    return torch.tensor(graph_wgts, dtype=torch.float64)


class OneIDSmallDataset(InMemoryDataset):
    def __init__(self, root="dataset", transform=None, pre_transform=None):
        self.original_root = root
        self.folder = osp.join(root, "one-id-small")
        self.data_ver = "v9"
        dict_edge_attr_dim = {
            "v2": 2,
            "v3": 5,
            "v4": 6,
            "v5": 5,
            "v6": 5,
            "v7": 2,
            "v8": 2,
            "v9": 6,
        }
        self.edge_attr_dim = dict_edge_attr_dim[self.data_ver]

        super(OneIDSmallDataset, self).__init__(self.folder, transform, pre_transform)
        fn_processed = self.processed_paths[0]
        print(f"Loading data from {fn_processed}")
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return f"data_2m_{self.data_ver}.pt"

    @property
    def split_idx_file_name(self):
        return f"split_dict_{self.data_ver}.pt"

    def download(self):
        pass

    def process(self):
        # data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names))
        # 1. initialize env: in odps
        from odps.types import Record
        import base64
        from odps.inter import enter

        # from odps.inter import setup
        # # [setup room](https://pyodps.readthedocs.io/zh_CN/latest/interactive.html#cl)
        # setup(''
        #       , ''
        #       , 'tcif_uuic_dev'
        #       , endpoint='http://service-corp.odps.aliyun-inc.com/api'
        #       , room='tcif_uuic_dev')
        # ossutil cp -r /mnt/workspace/kgg/qf/ggpt/data/OneID/one-id-small/processed/data.pt oss://dt-relation/dev/qifang/dataset/OneID/one-id-small/processed/
        def convert_odps_data(dp: Record):
            dp_val = dp.values
            x = torch.tensor(
                np.frombuffer(base64.b64decode(dp_val[2]), dtype=np.int64).reshape(
                    [-1, 1]
                )
            )
            edge_index = torch.tensor(
                np.frombuffer(base64.b64decode(dp_val[0]), dtype=np.int64).reshape(
                    [2, -1]
                )
            )
            edge_attr = torch.tensor(
                np.frombuffer(base64.b64decode(dp_val[1]), dtype=np.int64)
                .reshape([self.edge_attr_dim, -1])
                .T
            )
            a2d = torch.tensor(
                np.frombuffer(base64.b64decode(dp_val[4]), dtype=np.int64)
                .reshape([2, -1])
                .T
            )
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                a2d=a2d,
                num_nodes=len(x),
                key_type=torch.LongTensor([int(dp_val[7])]),
            )
            return graph, dp_val[8]

        if "o" not in globals():
            room = enter(room="tcif_uuic_dev")
            odps = o = room.odps
        tab_name = "uuic_taobao_a2d_with_d2d_group_graph_samples_20240121_v9"
        print(f"Converting odps table {tab_name} into graphs...")
        t = o.get_table(tab_name)

        # 2. process data
        split_dict = {"train": [], "valid": [], "test": []}
        data_list = []
        with t.open_reader() as reader:
            count = reader.count
            print(f"precessing {count} records ...")
            for i, record in tqdm(enumerate(reader[:count])):
                data, dtype = convert_odps_data(record)
                data_list.append(data)
                split_dict[dtype].append(i)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("Saving split dict ...")
        split_dict = {k: torch.tensor(v) for k, v in split_dict.items()}
        torch.save(split_dict, osp.join(self.root, self.split_idx_file_name))

        print("Collating ...")
        data, slices = self.collate(data_list)

        print("Saving data ...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        fn = osp.join(self.root, self.split_idx_file_name)
        print(f"loading idx split from {fn}")
        split_dict = torch.load(fn)
        return split_dict


class StructureDataset(InMemoryDataset):
    def __init__(
        self, root="dataset", transform=None, pre_transform=None, *, data_ver=None
    ):
        self.original_root = root
        self.folder = osp.join(root, "struct_cogn")
        self.data_ver = data_ver
        super(StructureDataset, self).__init__(self.folder, transform, pre_transform)
        fn_processed = self.processed_paths[0]
        print(f"Loading data from {fn_processed}")
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.idx_split_dict = None

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return f"data_{self.data_ver}.pt"

    @property
    def split_idx_file_name(self):
        return f"split_dict_{self.data_ver}.pt"

    def download(self):
        pass

    def process(self):
        if self.data_ver == "v00":
            data_list, split_dict = self._process_tri_reddit()
        elif self.data_ver in {"v01", "v02", "v03", "v04", "v05"}:
            data_list, split_dict = self._process_odps()
        else:
            data_list, split_dict = self._process_random_graphs()

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("Saving split dict ...")
        split_dict = {k: torch.tensor(v) for k, v in split_dict.items()}
        torch.save(split_dict, osp.join(self.root, self.split_idx_file_name))

        print("Collating ...")
        data, slices = self.collate(data_list)

        print("Saving data ...")
        torch.save((data, slices), self.processed_paths[0])

    def _process_random_graphs(self):
        from torch_geometric.utils import erdos_renyi_graph

        dict_ = {
            "v06": 0.15,
            "v07": 0.1,
            "v08": 0.05,
            "v09": 0.03,
            "v10": 0.01,
            "v18": 0.05,
            "v19": 0.03,
            "v20": 0.01,
        }

        def get_random_graphs(num_nodes, p):
            edge_index = erdos_renyi_graph(num_nodes, edge_prob=p, directed=False)
            graph = Data(edge_index=edge_index, num_nodes=num_nodes)
            return graph, round(num_nodes, -1)

        # 2. process data
        split_dict = {}
        data_list = []
        count = 3000000
        if self.data_ver in ("v18", "v19", "v20"):
            count = 3112359
        num_nodes_low, num_nodes_high = 4, 101
        vec_num_nodes = np.array(range(num_nodes_low, num_nodes_high))
        vec_num_nodes_cnt = vec_num_nodes
        vec_prob = vec_num_nodes_cnt / vec_num_nodes_cnt.sum()
        all_num_nodes = np.random.choice(
            vec_num_nodes, size=count, replace=True, p=vec_prob
        )
        prob = dict_[self.data_ver]
        print(
            f"precessing {count} records with num_nodes [{num_nodes_low}, {num_nodes_high}) and prob {prob} ..."
        )
        for i, num_nodes_ in tqdm(enumerate(all_num_nodes)):
            data, dtype = get_random_graphs(num_nodes_, prob)
            data_list.append(data)
            if not dtype in split_dict:
                split_dict[dtype] = []
            split_dict[dtype].append(i)

        if self.data_ver in ("v18", "v19", "v20"):
            srt = count
            data_list, split_dict, srt = _add_reddit_threads_data(
                data_list, split_dict, srt
            )
            data_list, split_dict, _ = _add_triangles_data(data_list, split_dict, srt)
        return data_list, split_dict

    def _process_odps(self):
        # data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names))
        # 1. initialize env: in odps
        from odps.types import Record
        import base64
        from odps.inter import enter

        # from odps.inter import setup
        # # [setup room](https://pyodps.readthedocs.io/zh_CN/latest/interactive.html#cl)
        # setup(''
        #       , ''
        #       , 'tcif_uuic_dev'
        #       , endpoint='http://service-corp.odps.aliyun-inc.com/api'
        #       , room='tcif_uuic_dev')
        # ossutil cp -r /mnt/workspace/kgg/qf/ggpt/data/OneID/one-id-small/processed/data.pt oss://dt-relation/dev/qifang/dataset/OneID/one-id-small/processed/
        def convert_odps_data(dp: Record):
            dp_val = dp.values
            edge_index = torch.tensor(
                np.frombuffer(base64.b64decode(dp_val[0]), dtype=np.int64).reshape(
                    [2, -1]
                )
            )
            graph = Data(
                edge_index=edge_index, num_nodes=torch.max(edge_index).item() + 1
            )
            # G = to_networkx(graph, to_undirected="upper").to_undirected()
            # graph.g = sum(nx.triangles(G).values()) // 3
            return graph, dp_val[2]

        if "o" not in globals():
            room = enter(room="tcif_uuic_dev")
            odps = o = room.odps
        tab_name = "tmp_ggpt_graph_cnts_convert_v5"
        print(f"Converting odps table {tab_name} into graphs...")
        t = o.get_table(tab_name)

        # 2. process data
        split_dict = {}
        data_list = []
        with t.open_reader() as reader:
            count = reader.count
            print(f"precessing {count} records ...")
            for i, record in tqdm(enumerate(reader[:count])):
                data, dtype = convert_odps_data(record)
                data_list.append(data)
                if not dtype in split_dict:
                    split_dict[dtype] = []
                split_dict[dtype].append(i)

        srt = count
        data_list, split_dict, srt = _add_reddit_threads_data(
            data_list, split_dict, srt
        )
        data_list, split_dict, _ = _add_triangles_data(data_list, split_dict, srt)

        return data_list, split_dict

    def _process_tri_reddit(self):
        split_dict = {}
        data_list = []
        srt = 0
        data_list, split_dict, srt = _add_reddit_threads_data(
            data_list, split_dict, srt
        )
        data_list, split_dict, _ = _add_triangles_data(data_list, split_dict, srt)
        return data_list, split_dict

    def get_idx_split(self):
        if self.idx_split_dict is None:
            fn = osp.join(self.root, self.split_idx_file_name)
            print(f"loading idx split from {fn}")
            self.idx_split_dict = torch.load(fn)
        return self.idx_split_dict


def _add_reddit_threads_data(data_list, split_dict, srt):
    from torch_geometric.datasets import TUDataset

    tu_dataset = TUDataset(root="../../data/TUDataset", name="reddit_threads")
    split_dict["reddit_threads"] = []
    for i, dp in tqdm(enumerate(tu_dataset, srt)):
        graph = Data(edge_index=dp.edge_index, num_nodes=dp.num_nodes)
        # G = to_networkx(graph, to_undirected="upper").to_undirected()
        # graph.g = sum(nx.triangles(G).values()) // 3
        data_list.append(graph)
        split_dict["reddit_threads"].append(i)
    print(f"top 10 reddit_threads index: {split_dict['reddit_threads'][:10]}")
    return data_list, split_dict, srt + len(tu_dataset)


def _add_triangles_data(data_list, split_dict, srt):
    from torch_geometric.datasets import TUDataset

    tu_dataset = TUDataset(root="../../data/TUDataset", name="TRIANGLES")
    split_dict["triangles-small"] = []
    split_dict["triangles-large"] = []
    for i, dp in tqdm(enumerate(tu_dataset, srt)):
        num_nodes = dp.x.shape[0]
        graph = Data(edge_index=dp.edge_index, num_nodes=num_nodes)
        # G = to_networkx(graph, to_undirected="upper").to_undirected()
        # graph.g = sum(nx.triangles(G).values()) // 3
        data_list.append(graph)
        if num_nodes <= 25:
            split_dict["triangles-small"].append(i)
        else:
            split_dict["triangles-large"].append(i)
    print(
        f"top 10 triangles-small: {split_dict['triangles-small'][:10]}, tot: {len(split_dict['triangles-small'])}"
    )
    print(
        f"top 10 triangles-large: {split_dict['triangles-large'][:10]}, tot: {len(split_dict['triangles-large'])}"
    )
    return data_list, split_dict, srt + len(tu_dataset)


class SpiceCircuitDataset(InMemoryDataset):
    def __init__(
        self, root="dataset", transform=None, pre_transform=None, *, data_ver=None
    ):
        self.original_root = root
        self.folder = osp.join(root, "spice_circuit")
        self.data_ver = data_ver
        super(SpiceCircuitDataset, self).__init__(self.folder, transform, pre_transform)
        fn_processed = self.processed_paths[0]
        print(f"Loading data from {fn_processed}")
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.idx_split_dict = None

    @property
    def raw_file_names(self):
        return "dummy.csv"

    @property
    def processed_file_names(self):
        return f"data.pt"

    @property
    def split_idx_file_name(self):
        return f"split_dict.pt"

    def download(self):
        pass

    def process(self):
        data_list, split_dict = self._process_spice_3350()

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("Saving split dict ...")
        split_dict = {k: torch.tensor(v) for k, v in split_dict.items()}
        torch.save(split_dict, osp.join(self.root, self.split_idx_file_name))

        print("Collating ...")
        data, slices = self.collate(data_list)

        print("Saving data ...")
        torch.save((data, slices), self.processed_paths[0])

    def _process_spice_3350(self):
        rt = "/Users/zhaoqifang/Downloads/graph-gpt/Dataset/"
        with open(os.path.join(rt, "graph-labels.csv"), "rb") as fp:
            labels = fp.readlines()
        labels = [int(x.strip()) for x in labels]
        print(f"labels: {len(labels)}")

        x_elements = []
        data_list = []

        start = 1
        end = 3502
        j = 0
        for i in range(start, end + 1):
            number = str(i)
            # Define file names
            netlist_file = os.path.join(rt, f"{number}/{number}.cir")
            port_file = os.path.join(rt, f"{number}/Port{number}.txt")
            if not os.path.isfile(netlist_file):
                # If it doesn't exist, skip the rest of this loop iteration
                print(f"{i}: Directory '{netlist_file}' does not exist. Skipping...")
                continue

            # Read netlist and ports
            netlist = spice2graph.read_netlist(netlist_file)
            ports = spice2graph.read_ports(port_file)

            # Build the connection matrix
            connection_matrix, net_connections = spice2graph.build_connection_matrix(
                netlist, ports
            )

            # convert into edge_index
            assert (
                connection_matrix.index == connection_matrix.columns
            ).all(), "row and col must be symmetric"
            adj_t = connection_matrix.to_numpy()
            assert (adj_t == adj_t.T).all(), "adj_t is NOT symmetric!"
            edge_index = torch.tensor(adj_t).nonzero().t().contiguous()

            # create x
            # update x's elements
            for ele in connection_matrix.index:
                if ele not in x_elements:
                    x_elements.append(ele)

            x = torch.tensor(
                [x_elements.index(ele) for ele in connection_matrix.index]
            ).reshape((-1, 1))
            label = labels[j]
            num_nodes = len(connection_matrix.index)

            data = Data(edge_index=edge_index, x=x, y=torch.tensor([label]))
            data.__num_nodes__ = num_nodes
            data_list.append(data)
            j += 1

        # save x's elements
        fn_x_elements = osp.join(self.folder, "x-elements.txt")
        with open(fn_x_elements, "wb") as fp:
            fp.writelines([f"{ele}\n".encode("utf-8") for ele in x_elements])
        print(f"Saving {len(x_elements)} x elements in fn {fn_x_elements}")

        # 2. process data
        split_dict = {}

        num_samples = len(data_list)
        seed = 42
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(num_samples, generator=g)

        split_dict["train"] = indices[: int(num_samples * 0.8)]
        split_dict["valid"] = indices[int(num_samples * 0.8) : int(num_samples * 0.9)]
        split_dict["test"] = indices[int(num_samples * 0.9) :]
        return data_list, split_dict

    def get_idx_split(self):
        if self.idx_split_dict is None:
            fn = osp.join(self.root, self.split_idx_file_name)
            print(f"loading idx split from {fn}")
            self.idx_split_dict = torch.load(fn)
        return self.idx_split_dict


if __name__ == "__main__":
    # import pyanitools as pya
    from ogb.lsc import PygPCQM4Mv2Dataset

    # dataset = PygPCQM4Mv2ExtraDataset(root="../../data/OGB")
    # dataset = PygPCQM4Mv2RdkitPosCCDataset(root="../../data/OGB")
    # dataset = PygPCQM4Mv2CCDataset(root="../../data/OGB")
    # dataset = PygPCQM4Mv2PosCCDataset(root="../../data/OGB")
    # dataset = StructureDataset(root="../../data/Struct", data_ver="v00")
    # dataset = OneIDSmallDataset(root="../../data/OneID")
    # dataset = PygPCQM4Mv2Dataset(root="../../data/OGB")
    # dataset = PygPCQM4Mv2PosDataset(root="../../data/OGB")
    # dataset = PygCEPDBDataset(root="../../data/OGB")
    # dataset = PygANI1Dataset(root="../../data/OGB")
    # dataset = PygZINCDataset(root="../../data/OGB", subset=11)
    dataset = SpiceCircuitDataset(root="../../data/Custom")
    print(dataset)
    data_ = dataset._data
    print(data_)
    print(data_.edge_index)
    print(data_.edge_index.shape)
    print(data_.x)
    print(data_.x.shape if hasattr(data_, "x") or data_.x else None)
    print(data_.edge_attr)
    print(
        data_.edge_attr.shape
        if hasattr(data_, "edge_attr") or data_.edge_attr
        else None
    )
    print(dataset[100])
    print(dataset[100].edge_index)
    print(dataset[100].y if hasattr(dataset[100], "y") else "NO y")
