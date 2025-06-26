import os
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from polylines import load_polylines, render_polylines
import glob

class PolylineDataset(Dataset):

    def __init__(
        self,
        folder_path: str,
        size: int = 256,
    ):
        self.folder_path = folder_path.strip()
        self.size = size
        
        # Find all .npy files recursively
        self.file_paths = self._find_npy_files()
        # sort by file name
        self.file_paths.sort()
        self.file_paths = self.file_paths[:1025]
        
        if len(self.file_paths) == 0:
            import pdb; pdb.set_trace()
            raise ValueError(f"No .npy files found in {folder_path}")
        
        print(f"Found {len(self.file_paths)} .npy files in {folder_path}")
    
    def _find_npy_files(self) -> List[Path]:
        return glob.glob(os.path.join(self.folder_path, "**/*.npy"), recursive=True)
    
    def _apply_rotation_augmentation(self, img: np.ndarray) -> np.ndarray:        
        # Random rotation: 0°, 90°, 180°, or 270°
        rotation_angle = random.choice([0, 90, 180, 270])
        
        if rotation_angle == 0:
            return img
        elif rotation_angle == 90:
            return np.rot90(img, k=1)
        elif rotation_angle == 180:
            return np.rot90(img, k=2)
        elif rotation_angle == 270:
            return np.rot90(img, k=3)
        
        return img
    
    def __len__(self) -> int:
        return len(self.file_paths) * 16
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        N = len(self.file_paths)
        idx = idx % N
        file_path = self.file_paths[idx]
        
        try:
            # Load polylines from file
            polylines = load_polylines(
                str(file_path),
                normalizing_scale=self.size,
                should_close=True
            )
            
            img = render_polylines(polylines, self.size)
            assert img.dtype != np.float32, f"Image is not uint8, got {img.dtype}. range: {img.min()} - {img.max()}"
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            img = render_polylines([], self.size)  
             
        img = img.astype(np.float32) / 255.0
        img = self._apply_rotation_augmentation(img).transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img.copy())  # Shape: (3, size, size)     
        return img_tensor, img_tensor.clone()

def create_polyline_dataloaders(
    folder_path: str,
    size: int,
    batch_size: int,
    split: float = 0.995,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    # Create dataset
    dataset = PolylineDataset(
        folder_path=folder_path,
        size=size
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(split * total_size)
    val_size = total_size - train_size
    
    print(f"Dataset splits: Train={train_size}, Val={val_size}")
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False
    )
    
    return train_loader, val_loader


def test_polyline_dataset():
    """Test function for the PolylineDataset."""
    # You'll need to update this path to point to your actual data
    test_folder = "/Users/souymodip/GIT/DATA_SETS/POLY1536_charac100/chunk_1"
    
    assert os.path.exists(test_folder), f"Test folder {test_folder} does not exist"
    # Test dataloaders
    train_loader, val_loader = create_polyline_dataloaders(
        folder_path=test_folder,
        size=256,
        batch_size=4,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print(f"Dataloaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    
    # Test a batch
    cond, target = next(iter(train_loader))
    print(f"Batch shape: {target.shape}")
    for i in range(target.shape[0]):
        plt.figure(figsize=(4, 4))
        plt.imshow(target[i].permute(1, 2, 0).cpu().numpy())
        plt.title(f"Sample {i}")
        plt.show()

if __name__ == "__main__":
    test_polyline_dataset() 
    