from dataclasses import dataclass, field
from typing import List, Optional, Any

ATTR_ASSIGNMENT_TYPES = {"first", "last", "random", "all", "mix"}

# Define the smallest, most nested classes first


@dataclass
class OdpsConfig:
    tables: str = ""  # ODPS input table names
    outputs: str = ""  # ODPS output table names
    edge_dim: Optional[int] = None  # shall be copied from tokenization cfg
    node_dim: Optional[int] = None  # shall be copied from tokenization cfg
    mode: Optional[str] = None
    # mode if not "train", every gpu will read whole table for valid/test


@dataclass
class DataConfig:
    data_dir: str = "../data/TUDataset"
    dataset: str = "reddit_threads"
    data_path: str = "reddit_threads"  # data_dir + data_path -> full path

    ensemble_datasets: Optional[List[str]] = None
    sampling: Optional[Any] = None

    return_valid_test: bool = False

    odps: OdpsConfig = field(default_factory=OdpsConfig)


@dataclass
class SemanticsSubConfig:
    discrete: Optional[str] = None
    dim: Optional[int] = None  # Optional because graph doesn't have it
    continuous: Optional[Any] = None
    ignored_val: Optional[Any] = None
    embed: Optional[str] = None
    embed_dim: Optional[int] = None


@dataclass
class SemanticsCommonConfig:
    reserved_token: List[str]
    numbers: List[str]


@dataclass
class SemanticsInstructionsConfig:
    enable: bool
    name: str
    func: List[Any] = field(default_factory=list)


@dataclass
class SemanticsConfig:
    node: SemanticsSubConfig
    edge: SemanticsSubConfig
    graph: SemanticsSubConfig
    common: SemanticsCommonConfig
    instructions: SemanticsInstructionsConfig
    attr_assignment: str = "first"  # available vals:: ATTR_ASSIGNMENT_TYPES
    attr_shuffle: bool = False


@dataclass
class StructureNxConfig:
    enable: bool
    func: List[Any] = field(default_factory=list)


@dataclass
class StructureNodeConfig:
    bos_token: str
    eos_token: str
    new_node_token: str
    node_scope: int
    scope_base: int
    cyclic: int


@dataclass
class StructureEdgeConfig:
    remove_edge_type_token: bool
    in_token: str
    out_token: str
    bi_token: str
    jump_token: str


@dataclass
class StructureGraphConfig:
    summary_token: str


@dataclass
class StructureCommonConfig:
    mask_token: str
    icl_token: str
    sep_token: str
    reserved_token: List[str]


@dataclass
class StructureConfig:
    nx: StructureNxConfig
    node: StructureNodeConfig
    edge: StructureEdgeConfig
    graph: StructureGraphConfig
    common: StructureCommonConfig


# This is the top-level dataclass that composes all the others
@dataclass
class TokenizationConfig:
    attr_world_identifier: str
    vocab_file: str
    label_tokens_to_pad: List[str]
    semantics: SemanticsConfig
    structure: StructureConfig
    tokenizer_class: str = (
        "StackedGSTTokenizer"  # GSTTokenizer|StackedGSTTokenizer|SPGSTTokenizer
    )
    data: DataConfig = field(default_factory=DataConfig)
    add_eos: bool = False
