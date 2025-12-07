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

def find_floor_and_box_planes(PC, threshold_floor, threshold_box, max_iter=10000):
    H, W, _ = PC.shape
    pts = PC.reshape(-1, 3) # Flatten

    # valid points only (z != 0)
    valid = pts[:, 2] != 0
    pts_valid = pts[valid]

    # 1) floor
    (n_floor, d_floor), inliers_floor = ransac_plane(
        pts_valid, threshold=threshold_floor, max_iter=max_iter
    )

    # map back to HxW (for visualization only)
    floor_mask = np.zeros(H*W, dtype=bool)
    floor_mask[np.where(valid)[0][inliers_floor]] = True
    floor_mask = floor_mask.reshape(H, W)

    # morphology cleanup (visualization only)
    floor_clean = binary_opening(floor_mask, structure=np.ones((3,3)))
    floor_clean = binary_closing(floor_clean, structure=np.ones((5,5)))
    floor_clean = binary_fill_holes(floor_clean)

    # ---------- Minimal changes start here ----------
    # (2) Fix normal direction so that "above" means positive signed distance.
    n_floor_u = n_floor / np.linalg.norm(n_floor)
    signed_valid = pts_valid @ n_floor_u - d_floor
    if np.median(signed_valid) < 0:
        n_floor, d_floor = -n_floor, -d_floor
        n_floor_u = -n_floor_u
        signed_valid = -signed_valid  # keep consistent

    # (1) Remove floor geometrically using distance to the plane; then keep only "above".
    signed_all = pts @ n_floor_u - d_floor
    floor_remove_eps = max(threshold_floor * 1.5, threshold_floor + 1e-9)

    keep_mask = valid & (np.abs(signed_all) > floor_remove_eps) & (signed_all > 0)
    pts_keep = pts[keep_mask]
    # ---------- Minimal changes end here ----------

    # 3) box top
    (n_top, d_top), inliers_top = ransac_plane(
        pts_keep, threshold=threshold_box, max_iter=max_iter
    )
    box_mask_all = np.zeros(H*W, dtype=bool)
    keep_idx = np.where(keep_mask)[0]
    box_mask_all[keep_idx[inliers_top]] = True
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

def save_overlay(image2d, mask, title, out_path):
    """Save an overlay of a binary mask on top of a 2D image."""
    plt.figure(figsize=(6,5))
    plt.imshow(image2d, cmap='gray')
    # show mask edges in a contrasting colormap with some transparency
    plt.imshow(np.ma.masked_where(~mask, mask), alpha=0.35, cmap='autumn')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def orthonormal_basis_from_normal(n):
    n = n / np.linalg.norm(n)
    # pick a vector not parallel to n
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v  # both unit, orthogonal, spanning the plane

def project_points_to_plane_uv(P, n, d):
    """
    P: (N,3) points on (or near) the plane
    plane: n·x = d  (with ||n||=1 recommended)
    Returns: U,V coordinates (N,2) in the plane frame and the (p0,u,v)
    """
    n = n / np.linalg.norm(n)
    # anchor point p0 on plane: along n from origin
    p0 = n * d
    u, v = orthonormal_basis_from_normal(n)
    Q = P - p0  # (N,3)
    UV = np.c_[Q @ u, Q @ v]  # (N,2)
    return UV, p0, u, v

def oriented_rect_from_points_2d(UV):
    """
    PCA-oriented rectangle on 2D points UV (N,2)
    Returns 4 corners in the same (U,V) coordinate frame, ordered around.
    """
    mu = UV.mean(axis=0)
    X = UV - mu
    C = (X.T @ X) / len(UV)
    w, V = np.linalg.eigh(C)        # ascending
    R = V[:, ::-1]                  # columns: principal axes (2x2)
    Y = X @ R
    mins = Y.min(axis=0); maxs = Y.max(axis=0)
    rect = np.array([
        [mins[0], mins[1]],
        [maxs[0], mins[1]],
        [maxs[0], maxs[1]],
        [mins[0], maxs[1]],
    ])
    # back to UV frame
    corners_uv = rect @ R.T + mu

    # order clockwise
    c = corners_uv.mean(axis=0)
    ang = np.arctan2(corners_uv[:,1]-c[1], corners_uv[:,0]-c[0])
    return corners_uv[np.argsort(ang)]

def corners3d_from_box_top(PC, box_top_mask, n_top, d_top):
    """Return 4 accurate 3D corners from the top mask using plane UV coordinates."""
    H, W, _ = PC.shape
    # gather top points (valid z)
    ys, xs = np.nonzero(box_top_mask)
    P = PC[ys, xs, :]              # (N,3)
    P = P[P[:,2] != 0]             # safety

    # project to plane 2D coords
    UV, p0, u, v = project_points_to_plane_uv(P, n_top, d_top)

    # (optional) small denoise in UV by removing extreme outliers
    # keep central quantile to avoid tiny speckles
    lo, hi = np.percentile(UV, [1, 99], axis=0)
    keep = (UV[:,0] >= lo[0]) & (UV[:,0] <= hi[0]) & (UV[:,1] >= lo[1]) & (UV[:,1] <= hi[1])
    UVc = UV[keep]

    # oriented rectangle in plane frame
    corners_uv = oriented_rect_from_points_2d(UVc)   # (4,2)

    # back to 3D: p = p0 + U*u + V*v
    corners_3d = np.array([p0 + c[0]*u + c[1]*v for c in corners_uv])
    return corners_3d  # (4,3)

# lengths (metric)
def lengths_from_corners_3d(P):
    d = lambda a,b: float(np.linalg.norm(P[a]-P[b]))
    e01, e12, e23, e30 = d(0,1), d(1,2), d(2,3), d(3,0)
    L = 0.5 * (e01 + e23); W = 0.5 * (e12 + e30)
    return (L, W) if L >= W else (W, L)

def save_corners_overlay(img, corners_rc, path, title):
    plt.figure(figsize=(6,5))
    plt.imshow(img, cmap='gray')
    r, c = corners_rc[:,0], corners_rc[:,1]
    plt.plot(np.r_[c, c[0]], np.r_[r, r[0]], '-')
    plt.scatter(c, r, s=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    
def corners_pixels_from_3d(PC, corners3d):
    H, W, _ = PC.shape
    pts = PC.reshape(-1,3)
    # naive nearest-neighbor in 3D; for speed you can downsample or use KD-tree
    idxs = []
    for p in corners3d:
        d2 = np.sum((pts - p)**2, axis=1)
        idxs.append(np.argmin(d2))
    idxs = np.array(idxs)
    rows = idxs // W
    cols = idxs %  W
    return np.c_[rows, cols]

def main():
    parser = argparse.ArgumentParser(description="Estimate box height from planes using RANSAC.")
    parser.add_argument("mat_path", type=Path, help="Path to the .mat file")
    parser.add_argument("--example", type=int, default=1, help="Example number in the .mat (default: 1)")
    parser.add_argument("--th_floor", type=float, default=0.01, help="RANSAC inlier threshold for floor (scene units)")
    parser.add_argument("--th_top", type=float, default=0.01, help="RANSAC inlier threshold for box top (scene units)")
    parser.add_argument("--save-viz", action="store_true", help="Save amplitude/cloud/mask visualizations")
    parser.add_argument("--sample-step", type=int, default=0, help="Downsample for point-cloud scatter (speed)")
    parser.add_argument("--max-itr", type=int, default=10000, help="Number of iterations to use in RANSAC Algoraithm")
    args = parser.parse_args()

    # In headless terminals, Agg avoids show() warnings
    import matplotlib
    matplotlib.use("Agg")

    # 1) Load data
    A, D, PC = load_example(args.mat_path, example_num=args.example)

    # 2) Optional visualizations
    if args.save_viz:
        plot_amplitude_image(A, title=f"Amplitude – Example {args.example}",
                             save_path=f"Results/example{args.example}_amplitude.png")
        try:
            plot_point_cloud(PC, color_by='z', sample_step=args.sample_step,
                             save_path=f"Results/example{args.example}_cloud.png")
        except Exception as e:
            print(f"[warn] Point cloud plot skipped: {e}")

    # 3) Find planes + masks
    (n_floor, d_floor, floor_mask), (n_top, d_top, box_top_mask) = \
    find_floor_and_box_planes(
        PC,
        threshold_floor=args.th_floor,
        threshold_box=args.th_top,
        max_iter=args.max_itr
    )

    # 4) Height
    h = box_height(n_floor, d_floor, n_top, d_top)
    print(f"Estimated box height (scene units): {h:.6f}")

    # Length & width
    Pcorners3D = corners3d_from_box_top(PC, box_top_mask, n_top, d_top)
    length, width = lengths_from_corners_3d(Pcorners3D)
    print(f"Length={length:.4f}, Width={width:.4f}")

    # 5) Save masks / overlays
    if args.save_viz:
        # raw masks already cleaned in find_floor_and_box_planes
        plt.imsave(f"Results/example{args.example}_floor_mask.png", floor_mask, cmap='gray')
        plt.imsave(f"Results/example{args.example}_boxtop_mask.png", box_top_mask, cmap='gray')

        # Overlays (use amplitude as the background)
        save_overlay(A, floor_mask, f"Floor mask – Example {args.example}",
                     f"Results/example{args.example}_floor_overlay.png")
        save_overlay(A, box_top_mask, f"Box-top mask – Example {args.example}",
                     f"Results/example{args.example}_boxtop_overlay.png")
        
        corners_rc = corners_pixels_from_3d(PC, Pcorners3D)
        save_corners_overlay(A, corners_rc,
            f"Results/example{args.example}_top_corners_refined.png",
            f"Top corners (3D-refined) – Example {args.example}")


if __name__ == "__main__":
    main()