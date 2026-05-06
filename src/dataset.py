from __future__ import annotations

import zipfile
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # Sử dụng thư viện chuẩn của PyTorch

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_CLASS_ORDER = [
    "Crazing", "Inclusion", "Patches", "Pitted_Surface", "Rolled-in_Scale", "Scratches",
]

CLASS_ALIASES = {
    "crazing": "Crazing", "inclusion": "Inclusion", "patches": "Patches",
    "pitted_surface": "Pitted_Surface", "pitted-surface": "Pitted_Surface",
    "rolled-in_scale": "Rolled-in_Scale", "rolled_in_scale": "Rolled-in_Scale",
    "rolled-in-scale": "Rolled-in_Scale", "scratches": "Scratches",
}

FILENAME_LABEL_PATTERN = re.compile(r"([A-Za-z_-]+)_\d+", re.IGNORECASE)

@dataclass
class SplitData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    input_dim: int 
    resolved_data_dir: str

class CNNDataset(Dataset):
    """
    Dataset cho CNN: Giữ nguyên cấu trúc không gian của ảnh.
    """
    def __init__(self, samples: list[tuple[Path, int]], transform: Callable | None = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        # Chuyển sang RGB vì các mô hình Pretrained như ResNet yêu cầu 3 kênh màu
        image = Image.open(path).convert("RGB") 
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_transforms(img_size: int, augment: bool = False):
    """
    Tạo các phép biến đổi ảnh cho CNN. 
    KHÔNG sử dụng flatten hay view(-1) ở đây[cite: 1].
    """
    # Chuẩn hóa theo thông số của ImageNet cho Transfer Learning[cite: 1]
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

# --- CÁC HÀM HỖ TRỢ GIỮ NGUYÊN TỪ FILE CŨ CỦA BẠN ---
def _normalize_label_name(name: str) -> str | None:
    key = name.strip().lower().replace(" ", "_")
    return CLASS_ALIASES.get(key)

def _ordered_class_names(names: list[str]) -> list[str]:
    unique = sorted(set(names))
    if set(unique).issubset(set(DEFAULT_CLASS_ORDER)):
        return [name for name in DEFAULT_CLASS_ORDER if name in unique]
    return unique

def _find_existing_data_path(data_path: Path) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    candidates = [data_path]
    if not data_path.is_absolute():
        candidates.extend([Path.cwd() / data_path, project_root / data_path])
    candidates.extend([project_root / "NEU-CLS.zip", project_root / "NEU-CLS"])
    for candidate in candidates:
        if candidate.exists(): return candidate
    raise FileNotFoundError(f"Không tìm thấy dữ liệu tại {data_path}")

def _extract_zip_if_needed(data_path: Path) -> Path:
    resolved_path = _find_existing_data_path(data_path)
    if resolved_path.is_dir(): return resolved_path
    if resolved_path.is_file() and resolved_path.suffix.lower() == ".zip":
        extract_root = resolved_path.parent / f"{resolved_path.stem}_extracted"
        if not (extract_root / ".extracted_ok").exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(resolved_path, "r") as zf:
                zf.extractall(extract_root)
            (extract_root / ".extracted_ok").write_text("ok")
        return extract_root
    return resolved_path

def _resolve_samples(data_dir: str | Path) -> tuple[list[tuple[Path, int]], list[str], Path]:
    root = _extract_zip_if_needed(Path(data_dir))
    # Quét thư mục con (Crazing/, ...)
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    
    samples: list[tuple[Path, int]] = []
    class_names = _ordered_class_names([_normalize_label_name(d.name) for d in class_dirs])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    for d in class_dirs:
        norm_name = _normalize_label_name(d.name)
        if norm_name in class_to_idx:
            for img_p in d.rglob("*"):
                if img_p.suffix.lower() in IMG_EXTENSIONS:
                    samples.append((img_p, class_to_idx[norm_name]))
    
    return samples, class_names, root

def create_dataloaders(
    data_dir: str | Path = "NEU-CLS.zip",
    img_size: int = 128, # CNN thường dùng 128 hoặc 224[cite: 1]
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    augment: bool = False,
    num_workers: int = 0,
) -> SplitData:
    samples, class_names, resolved_root = _resolve_samples(data_dir)
    
    # Chia tập dữ liệu
    labels = [s[1] for s in samples]
    train_idx, temp_idx = train_test_split(range(len(samples)), test_size=val_size+test_size, stratify=labels, random_state=random_state)
    
    val_rel_size = val_size / (val_size + test_size)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=1-val_rel_size, stratify=temp_labels, random_state=random_state)
    
    # Tạo Dataset
    train_ds = CNNDataset([samples[i] for i in train_idx], transform=get_transforms(img_size, augment))
    val_ds = CNNDataset([samples[i] for i in val_idx], transform=get_transforms(img_size, augment=False))
    test_ds = CNNDataset([samples[i] for i in test_idx], transform=get_transforms(img_size, augment=False))

    return SplitData(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        class_names=class_names,
        input_dim=img_size, # Với CNN, input_dim là kích thước cạnh ảnh[cite: 1]
        resolved_data_dir=str(resolved_root)
    )