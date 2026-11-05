from pathlib import Path

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from src.glaucoma_ml.data.manifest import LABEL_TO_INDEX
from src.glaucoma_ml.data.preprocessing import (
    build_image_transform,
    decode_and_resize,
)


class FundusDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        training: bool = False,
        image_size: int = 224,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.image_size = image_size
        self.transform = build_image_transform(training)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        image = decode_and_resize(image_path.read_bytes(), self.image_size)
        label = LABEL_TO_INDEX[str(row["label"])]
        return self.transform(image), label
