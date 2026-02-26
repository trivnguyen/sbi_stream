"""Matched filter dataset — placeholder for future implementation."""

from pathlib import Path
from typing import Union


def read_and_process_raw(data_dir: Union[str, Path], **kwargs):
    raise NotImplementedError("Matched filter dataset reading is not yet implemented.")


def read_processed(data_dir: Union[str, Path], **kwargs):
    raise NotImplementedError("Matched filter processed dataset reading is not yet implemented.")


def prepare_dataloaders(data, **kwargs):
    raise NotImplementedError("Matched filter dataloader preparation is not yet implemented.")


def prepare_test_dataloader(data, norm_dict, **kwargs):
    raise NotImplementedError("Matched filter test dataloader preparation is not yet implemented.")
