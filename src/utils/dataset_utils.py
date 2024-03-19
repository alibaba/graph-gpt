# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import os.path as osp
import shutil
import tarfile
import time
from typing import List, Optional
import numpy as np
import networkx as nx
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
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem

from multiprocessing import Pool


def mol2graph(mol):
    # refer to: https://github.com/lsj2408/Transformer-M/blob/main/Transformer-M/data/wrapper.py
    try:
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

        # positions
        positions = mol.GetConformer().GetPositions()

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)
        graph["position"] = positions

        return graph
    except:
        return None


class PygPCQM4Mv2PosDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=smiles2graph,
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
        self.smiles2graph = smiles2graph
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

        super(PygPCQM4Mv2PosDataset, self).__init__(
            self.folder, transform, pre_transform
        )

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

    def process(self):
        processes = 10
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        graph_pos_list = Chem.SDMolSupplier(
            osp.join(self.raw_dir, "pcqm4m-v2-train.sdf")
        )
        homolumogap_list = data_df["homolumogap"]
        num_3d = len(graph_pos_list)
        num_all = len(homolumogap_list)
        print(f"Totally {num_all} molecules, with {num_3d} having 3D positions!")

        print(
            f"Extracting 3D positions of {num_3d} molecules from SDF files for Training Data..."
        )
        train_data_with_position_list = []
        with Pool(processes=processes) as pool:
            iter = pool.imap(mol2graph, graph_pos_list)

            for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):
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

        smiles_list = data_df["smiles"][num_3d:].tolist()
        homolumogap_list = homolumogap_list[num_3d:].tolist()
        print(
            f"Converting SMILES strings of {len(smiles_list)} molecules into graphs..."
        )
        data_list = []
        with Pool(processes=processes) as pool:
            iter = pool.imap(smiles2graph, smiles_list)

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

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, "split_dict.pt"))
        )
        return split_dict


def smiles2graph_with_try(smiles_string):
    try:
        graph = smiles2graph(smiles_string)
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
        smiles2graph=smiles2graph,
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
        self.smiles2graph = smiles2graph
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

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "17294444"

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
        processes = 10
        data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names))
        smiles_list = data_df["SMILES_str"]
        homolumogap_list = data_df["e_gap_alpha"]

        print("Converting SMILES strings into graphs...")
        data_list = []
        with Pool(processes=processes) as pool:
            iter = pool.imap(smiles2graph_with_try, smiles_list)

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
        smiles2graph=smiles2graph,
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
        self.smiles2graph = smiles2graph
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
                iter = pool.imap(smiles2graph_with_try, smiles_list)

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
        smiles2graph=smiles2graph,
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
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "zinc")
        self.version = 1
        self.subset = subset

        # let's only consider at reference PH=7 first, so use *_p0.smi.gz
        self.urls = [
            f"http://zinc12.docking.org/db/bysubset/{subset}/{subset}_p0.smi.gz",
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
        return f"geometric_data_processed_{self.subset}.pt"

    def download(self):
        for url in self.urls:
            if decide_download(url):
                path = download_url(url, self.original_root)
                extract_zip(path, self.original_root)
                os.unlink(path)
            else:
                print("Stop download.")
                exit(-1)

    def process(self):
        processes = 10
        data_list = []
        print(
            f"processing {len(self.raw_file_names)} raw files: {self.raw_file_names} ..."
        )
        for raw_file in self.raw_file_names:
            data_df = pd.read_csv(
                osp.join(self.raw_dir, raw_file),
                sep=" ",
                header=None,
                names=["smiles", "zinc_id"],
            )
            smiles_list = data_df["smiles"]

            print(f"Converting SMILES strings from {raw_file} into graphs ...")

            with Pool(processes=processes) as pool:
                iter = pool.imap(smiles2graph_with_try, smiles_list)

                for i, graph in tqdm(enumerate(iter), total=len(smiles_list)):
                    try:
                        data = Data()

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
    assert hasattr(dataset, "x")
    x_idx = dataset.slices["x"]
    # cnt of nodes of all graphs
    cnt_nodes = x_idx[1:] - x_idx[:-1]
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
    def __init__(self, root="dataset", transform=None, pre_transform=None):
        self.original_root = root
        self.folder = osp.join(root, "struct_cogn")
        self.data_ver = "v5"
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
        # data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names))
        from torch_geometric.datasets import TUDataset

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
        tu_dataset = TUDataset(root="../../data/TUDataset", name="reddit_threads")
        split_dict["reddit_threads"] = []
        for i, dp in tqdm(enumerate(tu_dataset, srt)):
            graph = Data(edge_index=dp.edge_index, num_nodes=dp.num_nodes)
            # G = to_networkx(graph, to_undirected="upper").to_undirected()
            # graph.g = sum(nx.triangles(G).values()) // 3
            data_list.append(graph)
            split_dict["reddit_threads"].append(i)

        srt = count + len(tu_dataset)
        tu_dataset = TUDataset(root="../../data/TUDataset", name="TRIANGLES")
        split_dict["triangles"] = []
        for i, dp in tqdm(enumerate(tu_dataset, srt)):
            graph = Data(edge_index=dp.edge_index, num_nodes=dp.x.shape[0])
            # G = to_networkx(graph, to_undirected="upper").to_undirected()
            # graph.g = sum(nx.triangles(G).values()) // 3
            data_list.append(graph)
            split_dict["triangles"].append(i)

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
        if self.idx_split_dict is None:
            fn = osp.join(self.root, self.split_idx_file_name)
            print(f"loading idx split from {fn}")
            self.idx_split_dict = torch.load(fn)
        return self.idx_split_dict


if __name__ == "__main__":
    # import pyanitools as pya
    from ogb.lsc import PygPCQM4Mv2Dataset

    dataset = StructureDataset(root="../../data/Struct")
    # dataset = OneIDSmallDataset(root="../../data/OneID")
    # dataset = PygPCQM4Mv2Dataset(root="../../data/OGB")
    # dataset = PygPCQM4Mv2PosDataset(root="../../data/OGB")
    # dataset = PygCEPDBDataset(root="../../data/OGB")
    # dataset = PygANI1Dataset(root="../../data/OGB")
    # dataset = PygZINCDataset(root="../../data/OGB", subset=11)
    print(dataset)
    data_ = dataset._data
    print(data_)
    print(data_.edge_index)
    print(data_.edge_index.shape)
    print(data_.x)
    print(data_.x.shape if data_.x else None)
    print(data_.edge_attr)
    print(data_.edge_attr.shape if data_.edge_attr else None)
    print(dataset[100])
    print(dataset[100].edge_index)
    print(dataset[100].y if hasattr(dataset[100], "y") else "NO y")
