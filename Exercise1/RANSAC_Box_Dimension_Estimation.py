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

def compute_mlesac_cost(dists, threshold, gamma):
    """
    Compute MLESAC cost function: C = SUM ( p(d(s_i)) )
    where p = d(s_i) if d(s_i) < epslion, else p = Gamma (Error Constant)
    
    Parameters:
    dists : np.ndarray
        Array of distances from points to the model.
    threshold : float
        Inlier threshold epslion.
    gamma : float
        Outlier penalty constant (must be > threshold).
    
    Returns:
    float
        Total MLESAC cost (Sum of distances of inliers and gamma of outliers).
    """
    # Inliers contribute their distance, outliers contribute gamma
    cost = np.sum(np.where(dists < threshold, dists, gamma))
    return cost

def mlesac_plane(points, threshold=0.05, gamma=None, max_iter=5000, early_stop_ratio=0.85, rng=None):
    """
    MLESAC (Maximum Likelihood Estimation Sample Consensus) plane fitting.
    Uses distance-based cost function instead of binary inlier counting.
    
    Parameters:
    points : np.ndarray
        (N,3) valid 3D points.
    threshold : float
        Inlier distance threshold epslion.
    gamma : float, optional
        Outlier penalty constant. If None, defaults to threshold * 3.
        Must satisfy gamma > threshold.
    max_iter : int
        Maximum number of iterations.
    early_stop_ratio : float
        Early stopping if cost suggests this ratio of inliers.
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns:
    tuple
        ((n, d), inlier_mask) where n is normal, d is plane offset, mask is boolean array.
    """
    rng = np.random.default_rng() if rng is None else rng
    if gamma is None:
        gamma = threshold * 3.0
    if gamma <= threshold:
        gamma = threshold * 3.0  # ensure gamma > threshold
    
    N = points.shape[0]
    best_cost = np.inf
    best_model = None
    best_mask = None
    
    for _ in range(max_iter):
        idx = rng.choice(N, size=3, replace=False)
        model = fit_plane_from_3pts(points[idx])
        if model is None:
            continue
        n, d = model
        # distances = abs(n·x - d)
        dists = np.abs(points @ n - d)
        # Compute MLESAC cost
        cost = compute_mlesac_cost(dists, threshold, gamma)
        
        if cost < best_cost:
            best_cost = cost
            best_model = (n, d)
            best_mask = dists <= threshold
            # Early stopping: if cost suggests high inlier ratio
            expected_inlier_cost = threshold * early_stop_ratio * N
            if cost < expected_inlier_cost:
                break
    
    return best_model, best_mask  # (n,d), (N,)

def preemptive_ransac_plane(points, threshold=0.05, M=256, B=None, max_iter=None, 
                            scoring='ransac', gamma=None, early_stop_ratio=0.85, rng=None):
    """
    Preemptive RANSAC plane fitting.
    Generates M hypotheses upfront and evaluates them in batches, progressively
    pruning hypotheses using preemption function f(i) = (Using FLOOR function for all divisions) [M·2^(-[i/B])].
    
    Parameters:
    points : np.ndarray
        (N,3) valid 3D points.
    threshold : float
        Inlier distance threshold epslion.
    M : int
        Initial number of hypotheses to generate.
    B : int, optional
        Batch size (points per evaluation round). If None, defaults to N // 10.
    max_iter : int, optional
        Maximum iterations (not used in preemptive, kept for compatibility).
    scoring : str
        Scoring method: 'ransac' (inlier count) or 'mlesac' (distance-based cost).
    gamma : float, optional
        Outlier penalty for MLESAC scoring. If None and scoring='mlesac', uses threshold * 3.
    early_stop_ratio : float
        Early stopping threshold (not typically used in preemptive).
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns:
    tuple
        ((n, d), inlier_mask) where n is normal, d is plane offset, mask is boolean array.
    """
    rng = np.random.default_rng() if rng is None else rng
    N = points.shape[0]
    if B is None:
        B = max(10, N // 10)  # ensure reasonable batch size
    if scoring == 'mlesac':
        if gamma is None:
            gamma = threshold * 3.0
        if gamma <= threshold:
            gamma = threshold * 3.0
    
    # Step 1: Generate M hypotheses upfront
    hypotheses = []
    for _ in range(M):
        idx = rng.choice(N, size=3, replace=False)
        model = fit_plane_from_3pts(points[idx])
        if model is not None:
            hypotheses.append(model)
    
    if len(hypotheses) == 0:
        raise ValueError("Failed to generate any valid hypotheses")
    
    M_actual = len(hypotheses)
    scores = np.zeros(M_actual)  # Track score for each hypothesis
    active_indices = list(range(M_actual))  # Which hypotheses are still active
    
    # Shuffle point indices for random evaluation order
    point_indices = np.arange(N)
    rng.shuffle(point_indices)
    points_evaluated = 0
    
    # Step 2: Iterative evaluation with preemption
    while len(active_indices) > 1 and points_evaluated < N:
        # Get next batch of B points
        batch_size = min(B, N - points_evaluated)
        batch_indices = point_indices[points_evaluated:points_evaluated + batch_size]
        batch_points = points[batch_indices]
        
        # Evaluate all active hypotheses on this batch
        for h_idx in active_indices:
            n, d = hypotheses[h_idx]
            dists = np.abs(batch_points @ n - d)
            
            if scoring == 'ransac':
                # RANSAC scoring: count inliers
                inlier_count = int((dists <= threshold).sum())
                scores[h_idx] += inlier_count
            elif scoring == 'mlesac':
                # MLESAC scoring: distance-based cost (lower is better, so negate)
                cost = compute_mlesac_cost(dists, threshold, gamma)
                scores[h_idx] -= cost  # negate because we want to maximize (minimize cost)
            else:
                raise ValueError(f"Unknown scoring method: {scoring}")
        
        points_evaluated += batch_size
        
        # Calculate preemption: f(i) = [M·2^(-[i/B])]
        stage = points_evaluated // B
        num_keep = int(np.floor(M_actual * (2 ** (-stage))))
        num_keep = max(1, min(num_keep, len(active_indices)))
        
        # Sort active hypotheses by score and keep top ones
        # For RANSAC: higher score is better; for MLESAC: higher (less negative) is better
        sorted_active = sorted(active_indices, key=lambda i: scores[i], reverse=True)
        active_indices = sorted_active[:num_keep]
    
    # Step 3: Final evaluation on all points for the best remaining hypothesis
    best_idx = active_indices[0]
    n, d = hypotheses[best_idx]
    dists = np.abs(points @ n - d)
    inlier_mask = dists <= threshold
    
    return (n, d), inlier_mask  # (n,d), (N,)

def plane_mask_from_inliers(inlier_mask_flat, H, W): # Function to return the mask into binary image
    mask = np.zeros(H*W, dtype=bool)
    mask[:len(inlier_mask_flat)] = inlier_mask_flat
    return mask.reshape(H, W)

def find_floor_and_box_planes(PC, threshold_floor, threshold_box, max_iter=10000, 
                               algorithm='ransac', gamma=None, M=256, B=None):
    """
    Find floor and box top planes using specified RANSAC variant.
    
    Parameters
    ----------
    PC : np.ndarray
        Point cloud (H, W, 3).
    threshold_floor : float
        Inlier threshold for floor plane.
    threshold_box : float
        Inlier threshold for box top plane.
    max_iter : int
        Maximum iterations (for RANSAC/MLESAC).
    algorithm : str
        Algorithm to use: 'ransac', 'mlesac', 'preemptive', 'preemptive-mlesac'.
    gamma : float, optional
        MLESAC outlier penalty (required for mlesac/preemptive-mlesac).
    M : int
        Initial hypotheses for Preemptive RANSAC.
    B : int, optional
        Batch size for Preemptive RANSAC.
    
    Returns
    -------
    tuple
        ((n_floor, d_floor, floor_mask), (n_top, d_top, box_top_mask))
    """
    H, W, _ = PC.shape
    pts = PC.reshape(-1, 3) # Flatten

    # valid points only (z != 0)
    valid = pts[:, 2] != 0
    pts_valid = pts[valid]

    # 1) floor - select algorithm
    if algorithm == 'ransac':
        (n_floor, d_floor), inliers_floor = ransac_plane(
            pts_valid, threshold=threshold_floor, max_iter=max_iter
        )
    elif algorithm == 'mlesac':
        if gamma is None:
            gamma = threshold_floor * 3.0
        (n_floor, d_floor), inliers_floor = mlesac_plane(
            pts_valid, threshold=threshold_floor, gamma=gamma, max_iter=max_iter
        )
    elif algorithm == 'preemptive':
        (n_floor, d_floor), inliers_floor = preemptive_ransac_plane(
            pts_valid, threshold=threshold_floor, M=M, B=B, scoring='ransac'
        )
    elif algorithm == 'preemptive-mlesac':
        if gamma is None:
            gamma = threshold_floor * 3.0
        (n_floor, d_floor), inliers_floor = preemptive_ransac_plane(
            pts_valid, threshold=threshold_floor, M=M, B=B, scoring='mlesac', gamma=gamma
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # map back to HxW (for visualization only)
    floor_mask = np.zeros(H*W, dtype=bool)
    floor_mask[np.where(valid)[0][inliers_floor]] = True
    floor_mask = floor_mask.reshape(H, W)

    # morphology cleanup (visualization only)
    floor_clean = binary_opening(floor_mask, structure=np.ones((3,3)))
    floor_clean = binary_closing(floor_clean, structure=np.ones((5,5)))
    floor_clean = binary_fill_holes(floor_clean)

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

    # 3) box top - use same algorithm
    if algorithm == 'ransac':
        (n_top, d_top), inliers_top = ransac_plane(
            pts_keep, threshold=threshold_box, max_iter=max_iter
        )
    elif algorithm == 'mlesac':
        if gamma is None:
            gamma = threshold_box * 3.0
        (n_top, d_top), inliers_top = mlesac_plane(
            pts_keep, threshold=threshold_box, gamma=gamma, max_iter=max_iter
        )
    elif algorithm == 'preemptive':
        (n_top, d_top), inliers_top = preemptive_ransac_plane(
            pts_keep, threshold=threshold_box, M=M, B=B, scoring='ransac'
        )
    elif algorithm == 'preemptive-mlesac':
        if gamma is None:
            gamma = threshold_box * 3.0
        (n_top, d_top), inliers_top = preemptive_ransac_plane(
            pts_keep, threshold=threshold_box, M=M, B=B, scoring='mlesac', gamma=gamma
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
    parser = argparse.ArgumentParser(description="Estimate box height from planes using RANSAC variants.")
    parser.add_argument("mat_path", type=Path, help="Path to the .mat file")
    parser.add_argument("--example", type=int, default=1, help="Example number in the .mat (default: 1)")
    parser.add_argument("--th_floor", type=float, default=0.01, help="Inlier threshold for floor (scene units)")
    parser.add_argument("--th_top", type=float, default=0.01, help="Inlier threshold for box top (scene units)")
    parser.add_argument("--algorithm", type=str, default="ransac", 
                        choices=["ransac", "mlesac", "preemptive", "preemptive-mlesac"],
                        help="Algorithm to use: ransac, mlesac, preemptive, preemptive-mlesac (default: ransac)")
    parser.add_argument("--gamma", type=float, default=None, 
                        help="MLESAC outlier penalty (default: threshold * 3, only for mlesac/preemptive-mlesac)")
    parser.add_argument("--M", type=int, default=256, 
                        help="Preemptive RANSAC: initial number of hypotheses (default: 256)")
    parser.add_argument("--B", type=int, default=None, 
                        help="Preemptive RANSAC: batch size (default: N//10)")
    parser.add_argument("--save-viz", action="store_true", help="Save amplitude/cloud/mask visualizations")
    parser.add_argument("--sample-step", type=int, default=0, help="Downsample for point-cloud scatter (speed)")
    parser.add_argument("--max-itr", type=int, default=10000, help="Number of iterations for RANSAC/MLESAC")
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

    # 3) Find planes + masks using selected algorithm
    print(f"Using algorithm: {args.algorithm}")
    
    # Prepare algorithm-specific parameters
    algorithm_params = {
        'threshold_floor': args.th_floor,
        'threshold_box': args.th_top,
        'max_iter': args.max_itr,
        'algorithm': args.algorithm
    }
    
    # Add algorithm-specific parameters
    if args.algorithm in ['mlesac', 'preemptive-mlesac']:
        if args.gamma is None:
            args.gamma = args.th_floor * 3.0
            print(f"Using default gamma = {args.gamma:.6f} (threshold * 3)")
        algorithm_params['gamma'] = args.gamma
    
    if args.algorithm in ['preemptive', 'preemptive-mlesac']:
        algorithm_params['M'] = args.M
        algorithm_params['B'] = args.B
        print(f"Preemptive RANSAC: M={args.M}, B={args.B if args.B else 'auto'}")

    # Call find_floor_and_box_planes with appropriate parameters
    (n_floor, d_floor, floor_mask), (n_top, d_top, box_top_mask) = \
        find_floor_and_box_planes(PC, **algorithm_params)

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
        algo_suffix = f"_{args.algorithm}"
        plt.imsave(f"Results/example{args.example}_floor_mask{algo_suffix}.png", floor_mask, cmap='gray')
        plt.imsave(f"Results/example{args.example}_boxtop_mask{algo_suffix}.png", box_top_mask, cmap='gray')

        # Overlays (use amplitude as the background)
        save_overlay(A, floor_mask, f"Floor mask ({args.algorithm}) – Example {args.example}",
                     f"Results/example{args.example}_floor_overlay{algo_suffix}.png")
        save_overlay(A, box_top_mask, f"Box-top mask ({args.algorithm}) – Example {args.example}",
                     f"Results/example{args.example}_boxtop_overlay{algo_suffix}.png")
        
        corners_rc = corners_pixels_from_3d(PC, Pcorners3D)
        save_corners_overlay(A, corners_rc,
            f"Results/example{args.example}_top_corners_refined{algo_suffix}.png",
            f"Top corners ({args.algorithm}) – Example {args.example}")


if __name__ == "__main__":
    main()