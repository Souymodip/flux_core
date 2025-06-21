from dataclasses import dataclass
import numpy as np
from typing import List
import os
import cv2
import matplotlib.pyplot as plt


@dataclass
class Polyline:
    points: np.ndarray
    color: np.ndarray
    z: int

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        return self.points[index]
    
    def __setitem__(self, index, value):
        self.points[index] = value


def save_polylines(polylines: list[Polyline], filepath: str):
    """
    Save a list of Polyline objects to a binary file.
    
    Args:
        polylines: List of Polyline objects to save
        filepath: Path to save the file (will add .npy extension if not present)
    """
    if not filepath.endswith('.npy'):
        filepath += '.npy'
    
    # Convert the polylines to a serializable format
    serialized_polylines = []
    for poly in polylines:
        serialized_polylines.append({
            'points': poly.points,
            'color': poly.color,
            'z': poly.z
        })
    
    # Save using numpy's binary format
    np.save(filepath, serialized_polylines)
    # print(f"Saved {len(polylines)} polylines to {filepath}")

def close_polyline(polyline: Polyline) -> Polyline:
    """Optimized polyline closing with vectorized interpolation."""
    points = polyline.points
    last_point = points[-1]
    first_point = points[0]
    
    vector = first_point - last_point
    distance = np.linalg.norm(vector)
    
    if distance > 2:
        num_steps = int(distance)
        # Vectorized interpolation
        t_values = np.linspace(1/num_steps, 1, num_steps, endpoint=True)
        interpolated_points = last_point + vector * t_values[:, np.newaxis]
        polyline.points = np.vstack([points, interpolated_points])
    else:
        # Simple append
        polyline.points = np.vstack([points, first_point.reshape(1, -1)])
    
    return polyline

def remove_repeated_points(polyline: Polyline) -> Polyline:
    """Optimized removal of repeated points using vectorized operations."""
    points = polyline.points
    if len(points) <= 1:
        return polyline
    
    # Vectorized distance calculation
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    # Create mask for points to keep (first point + points with sufficient distance)
    keep_mask = np.concatenate([[True], distances > 1e-6])
    
    # Apply mask efficiently
    if not np.all(keep_mask):
        polyline.points = points[keep_mask]
    
    return polyline

def load_polylines(filepath: str, normalizing_scale: int|None=None, should_close: bool=False) -> List[Polyline]:
    """Optimized polyline loading with efficient filtering and processing."""
    if not filepath.endswith('.npy'):
        filepath += '.npy'
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load data once
    serialized_polylines = np.load(filepath, allow_pickle=True)
    
    # Pre-allocate list with estimated size
    polylines = []
    polylines_reserve = len(serialized_polylines)
    
    # Process polylines with early filtering
    for data in serialized_polylines:
        points = data['points']
        
        # Early length check before creating object
        if len(points) < 10:
            continue
        
        # Create polyline object
        polyline = Polyline(
            points=points,
            color=data['color'],
            z=data['z']
        )
        
        # Optimized point removal
        polyline = remove_repeated_points(polyline)
        
        # Check length again after point removal
        if len(polyline.points) < 10:
            continue
        
        # Optional closing
        if should_close:
            polyline = close_polyline(polyline)
        
        polylines.append(polyline)
    
    # Apply pose normalization once at the end if needed
    if normalizing_scale is not None and polylines:
        polylines = pose_polylines(polylines, normalizing_scale)
    
    return polylines


def polyline_to_mask(points, raster_size: int) -> np.ndarray:
    if len(points) < 3:
        import pdb; pdb.set_trace()
        return None
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)

    points = np.int32(points)        
    cv2.fillPoly(mask, [points], 255)
    return mask


def pose_polylines(polylines: List[Polyline], size: int) -> List[Polyline]:
    """Optimized pose normalization using vectorized operations."""
    if not polylines:
        return polylines
    
    # Collect all points efficiently
    all_points = np.vstack([poly.points[:, :2] for poly in polylines])
    
    # Vectorized min/max calculation
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    
    # Calculate transformation parameters
    width, height = max_xy - min_xy
    scale = size / max(width, height)
    
    # Center offset calculation
    dx = (size - width * scale) / 2
    dy = (size - height * scale) / 2
    dxy = np.array([dx, dy]) - min_xy * scale
    
    # Apply transformation to all polylines
    for polyline in polylines:
        polyline.points = polyline.points[:, :2] * scale + dxy
    
    return polylines


def render_polylines(polylines: list[Polyline], size: int, 
                     bg_color: np.ndarray=np.array([240, 250, 245])) -> np.ndarray:
    
    img = np.ones((size, size, 3), dtype=np.uint8) * bg_color[None, None, :]
    for i in range(len(polylines)):
        path = polylines[i].points[:, :2]
        color = (polylines[i].color * 255).astype(np.uint8)
        mask = polyline_to_mask(path, size)
        img[mask == 255] = color
    return img


def test():
    path = "/Users/souymodip/GIT/DATA_SETS/POLY1536_charac100/chunk_1/000005.npy"
    polylines = load_polylines(path, normalizing_scale=128)
    img = render_polylines(polylines, 128)
    plt.imshow(img)
    plt.show()
    
    
if __name__ == "__main__":
    test()
    