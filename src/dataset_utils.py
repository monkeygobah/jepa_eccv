from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset




@dataclass(frozen=True)
class CFCSplitSample:
    image_id: str
    side: str           
    path: Path


class CFCSplitDataset(Dataset):
    def __init__(self, root: Path, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.samples = self._index(self.root)
        self.transform = transform


    def _index(self, root: Path) -> list[CFCSplitSample]:
        out: list[CFCSplitSample] = []
        for p in sorted(root.rglob("*")):
            name = p.stem
            side = None
            if name.endswith("_OD"):
                side = "OD"
                image_id = name[:-3] 
            elif name.endswith("_OS"):
                side = "OS"
                image_id = name[:-3] 
            else:
                continue

            out.append(CFCSplitSample(image_id=image_id, side=side, path=p))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s