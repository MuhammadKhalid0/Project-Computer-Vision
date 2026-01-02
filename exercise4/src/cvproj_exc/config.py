from enum import Enum
from pathlib import Path
from typing import Final


class ReIdMode(Enum):
    IDENT = "ident"
    """The re-identification mode via classification."""
    CLUSTER = "cluster"
    """The re-identification mode via clustering."""


def enum_choices(enum_type: type[Enum]):
    return list(enum_type) + [e.value for e in enum_type]


class Config:
    PROJECT_DIR: Final[Path] = Path(__file__).parents[2]
    """The project directory path."""

    DATA_DIR: Final[Path] = PROJECT_DIR.joinpath("data")
    """The data directory path."""

    TRAIN_DATA: Final[Path] = DATA_DIR.joinpath("train_data")
    TEST_DATA: Final[Path] = DATA_DIR.joinpath("test_data")

    # specific files
    RESNET50: Final[Path] = DATA_DIR.joinpath("resnet50_128.onnx")
    CLUSTER_GALLERY: Final[Path] = DATA_DIR.joinpath("clustering_gallery.pkl")
    REC_GALLERY: Final[Path] = DATA_DIR.joinpath("recognition_gallery.pkl")

    EVAL_TRAIN_DATA: Final[Path] = DATA_DIR.joinpath("evaluation_train_data.pkl")
    EVAL_TEST_DATA: Final[Path] = DATA_DIR.joinpath("evaluation_test_data.pkl")

    CHAL_TRAIN_DATA: Final[Path] = DATA_DIR.joinpath("challenge_train_data.csv")
