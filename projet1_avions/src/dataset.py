"""Préparation des données FGVC-Aircraft : labels sur 3 niveaux, transformations, Dataset PyTorch."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LEVELS = ["manufacturer", "family", "variant"]
SPLITS = ["train", "val", "test"]


def load_labels(aircraft_dir: Path) -> pd.DataFrame:
    """Une ligne par image : image_id, ses 3 labels (texte) et son split."""

    def read(level, split):
        path = aircraft_dir / f"images_{level}_{split}.txt"
        rows = [line.split(" ", 1) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(rows, columns=["image_id", level]).set_index("image_id")[level]

    frames = []
    for split in SPLITS:
        df_split = pd.concat([read(level, split) for level in LEVELS], axis=1)
        df_split["split"] = split
        frames.append(df_split)
    return pd.concat(frames).reset_index()


def encode_labels(df: pd.DataFrame):
    """Ajoute une colonne <niveau>_idx (entier) par niveau et renvoie les noms de classes."""
    df = df.copy()
    classes = {}
    for level in LEVELS:
        classes[level] = sorted(df[level].unique())
        name_to_idx = {name: i for i, name in enumerate(classes[level])}
        df[f"{level}_idx"] = df[level].map(name_to_idx)
    return df, classes


def build_hierarchy(df: pd.DataFrame):
    """Tables parent : variant_to_family[v] = f et family_to_manufacturer[f] = m."""
    variant_to_family = df.groupby("variant_idx")["family_idx"].first().sort_index()
    family_to_manufacturer = df.groupby("family_idx")["manufacturer_idx"].first().sort_index()
    return torch.tensor(variant_to_family.values), torch.tensor(family_to_manufacturer.values)


# --- Transformations des images -----------------------------------------------

BANNER_PX = 20  # hauteur du bandeau de copyright en bas des images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_banner(img):
    """Retire le bandeau de copyright (nom du photographe) en bas de l'image."""
    w, h = img.size
    return img.crop((0, 0, w, h - BANNER_PX))


def build_transforms(train: bool, img_size: int = 224):
    """Transformations appliquées à une image PIL (bandeau déjà retiré)."""
    resize_size = int(img_size * 1.15)  # un peu plus grand que la taille finale, pour pouvoir recadrer
    if train:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),  # taille fixe, sans couper le nez ni la queue
            transforms.RandomCrop(img_size),                 # recadrage aléatoire
            transforms.RandomHorizontalFlip(),               # miroir gauche-droite, une fois sur deux
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(img_size),                     # recadrage central, toujours le même
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Inverse de Normalize, pour afficher un tenseur image avec matplotlib."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0)


# --- Dataset et DataLoader ----------------------------------------------------

class AircraftDataset(Dataset):
    """Renvoie (image, labels), avec labels = {"manufacturer": i, "family": j, "variant": k}."""

    def __init__(self, df: pd.DataFrame, aircraft_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(aircraft_dir) / "images"
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = crop_banner(Image.open(self.images_dir / f"{row.image_id}.jpg").convert("RGB"))
        if self.transform:
            img = self.transform(img)
        return img, {level: int(row[f"{level}_idx"]) for level in LEVELS}


def make_loader(df, aircraft_dir, split, train, img_size=224, batch_size=32, num_workers=4):
    """DataLoader d'un split. train=True active l'augmentation et le mélange."""
    subset = df[df["split"] == split]
    dataset = AircraftDataset(subset, aircraft_dir, build_transforms(train, img_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      persistent_workers=num_workers > 0)
