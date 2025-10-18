import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed to register 3D projection)
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label


def load_example(mat_path: Path, example_num: int = 1):
    mat_contents = loadmat(mat_path, simplify_cells=True)
    
    amp_key = f'amplitudes{example_num}'
    dist_key = f'distances{example_num}'
    cloud_key = f'cloud{example_num}'
    
    amplitudes = mat_contents.get(amp_key)
    distances = mat_contents.get(dist_key)
    cloud = mat_contents.get(cloud_key)
    
    if amplitudes is None or distances is None or cloud is None:
        raise KeyError(f"Example {example_num} not found in {mat_path}")
    
    return amplitudes, distances, cloud


def plot_point_cloud(cloud, color_by='z', sample_step=1, save_path=None):
    """
    Visualize a 3D point cloud (H, W, 3).

    Parameters
    ----------
    cloud : np.ndarray
        The point cloud array of shape (H, W, 3).
    color_by : str, optional
        Which channel to use for coloring ('x', 'y', or 'z'). Default: 'z'.
    sample_step : int, optional
        Downsample step to speed up plotting (e.g., 5 → plot every 5th point).
    save_path : str or Path, optional
        If given, saves the figure to this path instead of showing it.
    """
    if cloud.ndim != 3 or cloud.shape[2] != 3:
        raise ValueError("Input cloud must have shape (H, W, 3)")

    # Flatten (H, W, 3) → (H*W, 3)
    pts = cloud.reshape(-1, 3)

    # Split into X, Y, Z
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Choose color channel
    color_map = {'x': x, 'y': y, 'z': z}
    color = color_map.get(color_by, z)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, s=0.5, c=color, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')

    plt.tight_layout()
    plt.colorbar(sc, ax=ax, label=f'{color_by.upper()} value')

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved point cloud visualization to {save_path}")
    else:
        plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_amplitude_image(image, title="Amplitude Image", cmap='gray', save_path=None):
    """
    Display or save an amplitude (or intensity) image.

    Parameters
    ----------
    image : np.ndarray
        2D array (H, W) representing the amplitude/intensity image.
    title : str, optional
        Title for the plot.
    cmap : str, optional
        Matplotlib colormap to use. Default: 'gray'.
    save_path : str or Path, optional
        If given, saves the figure to this path instead of showing it.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D (H, W)")

    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar(label='Intensity')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved amplitude image to {save_path}")
    else:
        plt.show()


def fit_plane_from_3pts(p):
    # p: (3,3) points
    v1 = p[1] - p[0]
    v2 = p[2] - p[0]
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return None  # nearly collinear
    n = n / n_norm               # unit normal
    d = np.dot(n, p[0])          # plane: n·x = d
    return n, d

def ransac_plane(points, threshold=0.05, max_iter=5000, early_stop_ratio=0.85, rng=None):
    """
    points: (N,3) valid 3D points
    threshold: inlier distance threshold from the plane
    returns: (n, d, inlier_mask_bool)
    """
    rng = np.random.default_rng() if rng is None else rng
    N = points.shape[0]
    best_inliers = 0
    best_model = None
    best_mask = None

    # Precompute for speed
    for _ in range(max_iter):
        idx = rng.choice(N, size=3, replace=False)
        model = fit_plane_from_3pts(points[idx])
        if model is None:
            continue
        n, d = model
        # distances = |n·x - d|
        dists = np.abs(points @ n - d)
        mask = dists <= threshold
        count = int(mask.sum())
        if count > best_inliers:
            best_inliers = count
            best_model = (n, d)
            best_mask = mask
            if best_inliers >= early_stop_ratio * N:
                break

    return best_model, best_mask  # (n,d), (N,)

def plane_mask_from_inliers(inlier_mask_flat, H, W): # Function to return the mask into binary image
    mask = np.zeros(H*W, dtype=bool)
    mask[:len(inlier_mask_flat)] = inlier_mask_flat
    return mask.reshape(H, W)

def find_floor_and_box_planes(PC, threshold_floor, threshold_box, max_iter=5000):
    H, W, _ = PC.shape
    pts = PC.reshape(-1, 3) #Flatten

    # valid points only (z != 0) # as per the instructions
    valid = pts[:, 2] != 0
    pts_valid = pts[valid]

    # 1) floor
    (n_floor, d_floor), inliers_floor = ransac_plane(pts_valid, threshold=threshold_floor, max_iter=max_iter)
    floor_mask_valid = inliers_floor
    # map back to HxW
    floor_mask = np.zeros(H*W, dtype=bool)
    floor_mask[np.where(valid)[0][floor_mask_valid]] = True
    floor_mask = floor_mask.reshape(H, W)

    # morphology cleanup (tune structure size)
    floor_clean = binary_opening(floor_mask, structure=np.ones((3,3)))
    floor_clean = binary_closing(floor_clean, structure=np.ones((5,5)))
    floor_clean = binary_fill_holes(floor_clean)

    # 2) remove floor points
    keep_mask = (~floor_clean).reshape(-1)
    keep_mask = keep_mask & valid

    pts_keep = pts[keep_mask]

    # 3) box top
    (n_top, d_top), inliers_top = ransac_plane(pts_keep, threshold=threshold_box, max_iter=max_iter)
    box_mask_all = np.zeros(H*W, dtype=bool)
    box_mask_all[np.where(keep_mask)[0][inliers_top]] = True
    box_mask = box_mask_all.reshape(H, W)

    # largest connected component on box mask
    lab, num = label(box_mask)
    if num > 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep_label = sizes.argmax()
        box_top_cc = (lab == keep_label)
    else:
        box_top_cc = box_mask

    return (n_floor, d_floor, floor_clean), (n_top, d_top, box_top_cc)

def box_height(n_floor, d_floor, n_top, d_top):
    # Ensure normals point roughly the same way
    if np.dot(n_floor, n_top) < 0:
        n_top, d_top = -n_top, -d_top
    # distance between parallel planes with unit normals
    return abs(d_top - d_floor)
