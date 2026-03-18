from abc import ABC, abstractmethod
from typing import Dict, Type


class TrainingMode(ABC):
    """Strategy interface for pretrain vs finetune training modes.

    Subclasses implement mode-specific logic while the TrainingPipeline
    handles shared orchestration (config extraction, distributed setup,
    model creation, checkpoint handling, cleanup).
    """

    @property
    @abstractmethod
    def dict_models(self) -> Dict[str, Type]:
        """Model class registry mapping model_type strings to classes."""
        ...

    @property
    def skip_keys(self) -> bool:
        """Whether to skip score-related keys when loading pretrained checkpoint.
        PT=True (default), FT=False.
        """
        return True

    def allow_resume(self) -> bool:
        """Whether to allow resuming from an existing checkpoint.
        FT overrides to return False when eval_only.
        """
        return True

    def allow_save_config(self) -> bool:
        """Whether to save model config to output_dir.
        FT overrides to return False when eval_only.
        """
        return True

    @property
    def final_config_filename(self) -> str:
        """Filename for the final config saved during cleanup.
        PT uses 'config_final.yaml' (default), FT uses 'config.yaml'.
        """
        return "config_final.yaml"

    @abstractmethod
    def update_config(self, pipeline) -> None:
        """Mode-specific config updates applied before any setup."""
        ...

    @abstractmethod
    def prepare_data(self, pipeline) -> None:
        """Full data pipeline: tokenizer config, dataset reading, vocab,
        tokenizer init, sampler creation, schedule updates, model config.

        Must set on pipeline: gtokenizer, tokenizer_cls, tokenizer_config, config (legacy).
        """
        ...

    @abstractmethod
    def post_model_setup(self, pipeline) -> bool:
        """Post-model-creation setup (e.g., freeze layers, early exit checks).
        Returns True to signal early exit (eval_only / infer_only).
        """
        ...

    @abstractmethod
    def setup_optimizer(self, pipeline) -> None:
        """Create optimizer (DeepSpeed or DDP), initialize EMA model.
        Must set on pipeline: opt_stats, device.
        """
        ...

    @abstractmethod
    def setup_training(self, pipeline) -> None:
        """Initialize logging config, collator, eval loaders, TB writer,
        pre-training evaluation, and TrainingStats.
        Must set on pipeline: train_stats, tb_writer.
        """
        ...

    @abstractmethod
    def run_training(self, pipeline) -> None:
        """Execute the training loop."""
        ...

    @abstractmethod
    def run_training(self, pipeline) -> None:
        """Execute the training loop."""
        ...
