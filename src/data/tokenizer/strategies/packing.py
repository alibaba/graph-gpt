"""Sequence packing strategy for efficient training."""

import random
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from torch.utils.data import IterableDataset, Dataset

if TYPE_CHECKING:
    from ..types import TokenizationOutput


class SequencePacker:
    """Packs multiple short sequences into a single long sequence."""

    def __init__(
        self,
        mpe: int,  # max position embeddings
        dataset,
        sampler=None,
        random_ratio: float = 1.0,
        eos_token: str = "<eos>",
        label_pad_token: str = "<label_pad>",
    ):
        self.mpe = mpe
        self.dataset = dataset
        self.sampler = sampler
        self.random_ratio = random_ratio
        self.eos_token = eos_token
        self.label_pad_token = label_pad_token

        if isinstance(self.dataset, IterableDataset):
            self.dataset = iter(self.dataset)

    def pack(
        self,
        token_res: "TokenizationOutput",
        previous_idx: int,
        tokenize_fn,
    ) -> Tuple[List, List, Optional[List], List[int]]:
        """
        Pack multiple tokenized sequences into one.

        Args:
            token_res: Initial tokenization result
            previous_idx: Index of previous sample (for non-random sampling)
            tokenize_fn: Function to tokenize a new graph

        Returns:
            Tuple of (ls_tokens, ls_labels, ls_embed, ls_len)
        """
        ls_tokens = list(token_res.ls_tokens)
        ls_labels = list(token_res.ls_labels)
        ls_embed = list(token_res.ls_embed) if token_res.ls_embed else token_res.ls_embed

        token_components = self._get_token_components(ls_tokens)
        token_len = len(ls_tokens) + 1
        ls_len = [token_len]

        while token_len < self.mpe:
            idx, new_graph = self._sample_next(previous_idx)
            new_token_res = tokenize_fn(new_graph)
            new_ls_tokens = new_token_res.ls_tokens
            new_ls_labels = new_token_res.ls_labels
            new_ls_embed = new_token_res.ls_embed

            seps, label_seps, embed_seps = self._create_separators(
                token_components, ls_embed
            )

            if token_len + len(seps) + len(new_ls_tokens) >= self.mpe:
                break

            ls_tokens.extend(seps)
            ls_tokens.extend(new_ls_tokens)
            ls_labels.extend(label_seps)
            ls_labels.extend(new_ls_labels)

            if ls_embed:
                ls_embed.extend(embed_seps)
                ls_embed.extend(new_ls_embed)

            previous_idx = idx
            token_len = len(ls_tokens) + 1
            ls_len.append(token_len)

        return ls_tokens, ls_labels, ls_embed, ls_len

    def _sample_next(self, previous_idx: int):
        """Sample next graph from dataset."""
        if isinstance(self.dataset, Dataset):
            if random.uniform(0, 1.0) <= self.random_ratio:
                idx = (
                    self.dataset.get_random_sample_idx()
                    if hasattr(self.dataset, "get_random_sample_idx")
                    else random.choice(self.sampler)
                )
            else:
                idx = previous_idx
            return idx, self.dataset[idx][1]
        else:
            return 0, next(self.dataset)[1]

    def _get_token_components(self, ls_tokens: List) -> int:
        """Determine if tokens are 1D or 2D (stacked)."""
        if ls_tokens and isinstance(ls_tokens[0], List):
            return len(ls_tokens[0])
        return 0

    def _create_separators(
        self, token_components: int, ls_embed: Optional[List]
    ) -> Tuple[List, List, List]:
        """Create separator tokens between packed sequences."""
        if token_components == 0:
            seps = [self.eos_token]
            label_seps = [self.label_pad_token]
        else:
            seps = [[self.eos_token] * token_components]
            label_seps = [[self.label_pad_token] * token_components]

        embed_seps = []
        if ls_embed:
            dim = len(ls_embed[0])
            embed_seps = np.zeros((1, dim), dtype=np.float32).tolist()

        return seps, label_seps, embed_seps
