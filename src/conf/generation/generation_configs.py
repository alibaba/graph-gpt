# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# copied and modified from https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/generation_utils.py
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from transformers import __version__

# It's good practice to keep the original Transformers class for type checking if needed,
# but our Hydra config object will be the primary data-carrier.
from transformers.generation.configuration_utils import GenerationConfig


@dataclass
class GenerationConfig:
    """
    Configuration for masked diffusion generation, designed for use with Hydra.

    This class holds all parameters for the generation process. It is instantiated
    by Hydra from a YAML configuration file.
    """

    # The _target_ field is for Hydra's internal use and won't be part of the class __init__.
    # We don't need to define it here.

    # --- Diffusion Specific Parameters ---
    alg: str = "origin"
    alg_temp: Optional[float] = None
    steps: int = 512
    eps: float = 1e-3
    parallel_gen: bool = False

    # --- Sampling Parameters ---
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    # --- Length Control ---
    max_length: int = 20
    max_new_tokens: Optional[int] = None

    # --- Output Control ---
    num_return_sequences: int = 1
    return_dict_in_generate: bool = False
    output_history: bool = False

    # --- Special Token IDs ---
    mask_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # --- Wild card and Metadata (for compatibility with original class) ---
    # Use field(default_factory=dict) for mutable defaults like dictionaries.
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)

    # These are mostly for tracking and compatibility with Hugging Face Hub.
    _from_model_config: bool = False
    _commit_hash: Optional[str] = None
    transformers_version: str = field(default=__version__)

    def __post_init__(self):
        """
        This method is called by dataclasses after the object is created.
        It's the perfect place for validation logic.
        """
        self.validate()

    def validate(self):
        """
        Validates the configuration parameters. This is the same method
        as in the original class.
        """
        if self.alg not in ["origin", "maskgit_plus", "topk_margin", "entropy"]:
            warnings.warn(
                f"Unknown algorithm '{self.alg}' specified. "
                f"Valid options are: 'origin', 'maskgit_plus', 'topk_margin', 'entropy'"
            )

        # You can add any other validation logic here.
        if self.temperature < 0:
            raise ValueError("`temperature` must be non-negative.")
        if self.steps <= 0:
            raise ValueError("`steps` must be positive.")
