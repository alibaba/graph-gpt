from .graphgpt.modeling_graphgpt import (
    GraphGPTPretrainBase,
    GraphGPTPosPred,
    GraphGPTTaskModel,
    GraphGPTDoubleHeadsModel,
    GraphGPTDenoisingRegressionDoubleHeadsModel,
)
from .graphgpt.configuration_graphgpt import GraphGPTConfig, convert_to_legacy_config

__all__ = [
    "convert_to_legacy_config",
    # below is fine-tune models
    "GraphGPTDenoisingRegressionDoubleHeadsModel",
    "GraphGPTDoubleHeadsModel",
    "GraphGPTTaskModel",
    # below is pre-train models
    "GraphGPTPosPred",
    "GraphGPTPretrainBase",
    "GraphGPTConfig",
]
