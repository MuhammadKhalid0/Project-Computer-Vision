import numpy as np
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import argparse
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

# Exercise 9: HDR from JPG images using camera response curve estimation
# Based on Debevec & Malik algorithm (1997)
# Reference: "Recovering High Dynamic Range Radiance Maps from Photographs"
# SIGGRAPH 1997

# Load images - A45A6923.JPG through A45A6934.JPG (12 images)
# Each image has half the exposure time of the previous one
image_files = [f"A45A69{i:02d}.JPG" for i in range(23, 35)]

# Exposure times: t_j = 1 / (2^j) where j = 0, 1, ..., 11
#exposure times are relative, so t_0 = 1.0 (longest), t_1 = 0.5, t_2 = 0.25, etc.
exposure_times = np.array([1.0 / (2**j) for j in range(12)], dtype=np.float32) #relative exposure times

print(f"Loading {len(image_files)} images...")
images = []
for fname in image_files:
    img = imageio.imread(fname).astype(np.float32)
    images.append(img)
    print(f"  Loaded {fname}, shape: {img.shape}")

images = np.array(images)  # shape: (num_images, H, W, 3)
print(f"Images loaded. Shape: {images.shape}")


def sample_pixels(images, n_samples_per_image=100, z_min=10, z_max=245):
    """
    Sample pixels from images for response curve estimation.
    Avoid very dark (< z_min) and very bright (> z_max) pixels.
    
    Returns:
        samples: list of (image_idx, pixel_idx, z_value) tuples
        pixel_locations: array of (row, col) for each sampled pixel
    """
    H, W = images.shape[1], images.shape[2]
    num_images = images.shape[0]
    
    # Sample random pixel locations
    # Using fixed seed so results are reproducible (tried different values, 42 works)
    np.random.seed(42)
    n_total = n_samples_per_image * num_images #100*12 = 1200 pixels
    
    # Random pixel locations
    rows = np.random.randint(0, H, size=n_total)
    cols = np.random.randint(0, W, size=n_total)
    
    samples = []
    pixel_locations = []
    
    for idx in range(n_total):
        r, c = rows[idx], cols[idx]
        
        # Check if this pixel is well-exposed in at least 2 images
        # (not too dark, not too bright)
        pixel_values = images[:, r, c, :]  # (num_images, 3) - RGB values #12x3 = 36 values
        
        # Use average across RGB channels (or could use luminance)
        z_avg = np.mean(pixel_values, axis=1)  # (num_images,) #one intensity value for each image
        
        # Count how many images have this pixel in good range
        good_count = np.sum((z_avg >= z_min) & (z_avg <= z_max))
        
        if good_count >= 2:  # Need at least 2 good exposures
            pixel_locations.append((r, c))
            
            # Add samples for each image
            for img_idx in range(num_images):
                z_val = int(np.mean(pixel_values[img_idx]))  # average RGB
                if z_min <= z_val <= z_max:
                    samples.append((img_idx, len(pixel_locations) - 1, z_val))
    
    print(f"Sampled {len(samples)} pixel observations from {len(pixel_locations)} locations")
    return samples, np.array(pixel_locations)


def build_response_system(samples, exposure_times, lambda_smooth=100, z_min=0, z_max=255):
    """
    Build the linear system for Debevec-Malik algorithm.
    
    We solve for:
    - g(0), g(1), ..., g(255) - response curve at each pixel value
    - E'_0, E'_1, ..., E'_N - log irradiance for each sampled pixel
    
    Objective:
    Minimize: sum[g(Z_ij) - E'_i - log(t_j)]^2 + lambda * sum[g''(z)]^2
    
    Returns:
        A: sparse matrix (coefficients)
        b: right-hand side vector
    """
    num_samples = len(samples)
    num_pixels = len(set(s[1] for s in samples))  # unique pixel locations
    
    # Number of unknowns:
    # - 256 for g(0) through g(255)
    # - num_pixels for E'_i (log irradiance)
    num_unknowns = 256 + num_pixels
    
    # Build mapping from pixel location to E' index
    pixel_to_eidx = {}
    eidx = 0
    for _, pix_idx, _ in samples:
        if pix_idx not in pixel_to_eidx:
            pixel_to_eidx[pix_idx] = eidx
            eidx += 1
    
    # Initialize sparse matrix
    # Rows: num_samples (data) + 254 (smoothness) + 1 (normalization)
    A = lil_matrix((num_samples + 254 + 1, num_unknowns))
    b = np.zeros(num_samples + 254 + 1)
    
    row_idx = 0
    
    # Data fitting equations: g(Z_ij) - E'_i - log(t_j) = 0
    for img_idx, pix_idx, z_val in samples:
        # g(Z_ij) term: coefficient 1 at column z_val
        A[row_idx, z_val] = 1.0
        
        # E'_i term: coefficient -1 at column (256 + e_idx)
        e_idx = pixel_to_eidx[pix_idx]
        A[row_idx, 256 + e_idx] = -1.0
        
        # Right-hand side: log(t_j)
        b[row_idx] = np.log(exposure_times[img_idx] + 1e-10)  # avoid log(0)
        
        row_idx += 1
    
    # Smoothness constraints: g''(z) = g(z+1) - 2*g(z) + g(z-1) ≈ 0
    # We want to minimize the second derivative
    for z in range(1, 255):  # z from 1 to 253 (can't compute g'' at boundaries)
        A[row_idx, z - 1] = lambda_smooth * 1.0   # g(z-1)
        A[row_idx, z] = lambda_smooth * (-2.0)    # -2*g(z)
        A[row_idx, z + 1] = lambda_smooth * 1.0  # g(z+1)
        b[row_idx] = 0.0  # we want g''(z) = 0
        row_idx += 1
    
    # Normalization constraint: g(128) = 0 (middle gray)
    # This fixes the scale ambiguity - without this, solution is not unique
    # (tried different values, 128 seems standard)
    A[row_idx, 128] = 1.0
    b[row_idx] = 0.0
    
    print(f"Built system: {A.shape[0]} equations, {num_unknowns} unknowns")
    return A.tocsr(), b


def solve_response_curve(A, b):
    """
    Solve the linear system Ax = b for the response curve.
    
    Returns:
        g: array of g(0) through g(255) - the log inverse response curve
    """
    print("Solving linear system...")
    
    # Use sparse least squares solver
    result = lsqr(A, b, atol=1e-6, btol=1e-6)
    x = result[0]
    
    # Extract g(z) values (first 256 elements)
    g = x[:256]
    
    print(f"Solved. g(0)={g[0]:.3f}, g(128)={g[128]:.3f}, g(255)={g[255]:.3f}")
    
    return g


def estimate_response_curve_custom(images, exposure_times, lambda_smooth=100):
    """
    Estimate camera response curve using our custom Debevec-Malik implementation.
    
    This implements the algorithm from the paper:
    - Sample pixels across different exposures
    - Build linear system: g(Z_ij) = E'_i + log(t_j) with smoothness constraint
    - Solve using least squares
    
    Tried different lambda values - 100 seems to work well (not too smooth, not too jagged)
    
    Returns:
        g: log inverse response curve [g(0), ..., g(255)]
        f_inv: inverse response curve (maps pixel value -> linear irradiance)
    """
    print("\n=== Custom Debevec-Malik Implementation ===")
    
    # Sample pixels
    samples, pixel_locations = sample_pixels(images, n_samples_per_image=150)
    
    if len(samples) < 100:
        print("Warning: Not enough samples, trying with more...")
        samples, pixel_locations = sample_pixels(images, n_samples_per_image=200)
    
    # Build linear system
    A, b = build_response_system(samples, exposure_times, lambda_smooth=lambda_smooth)
    
    # Solve
    g = solve_response_curve(A, b)
    
    # Convert g to f_inv: f_inv(z) = exp(g(z))
    # This maps pixel value z -> linear irradiance
    f_inv = np.exp(g)
    
    # Normalize so f_inv(128) = 1.0 (middle gray -> unit irradiance)
    # This is just for scaling, doesn't affect HDR combination
    f_inv = f_inv / (f_inv[128] + 1e-10)
    
    return g, f_inv


def estimate_response_curve_opencv(images, exposure_times):
    """
    Estimate camera response curve using OpenCV's implementation.
    
    Returns:
        response: OpenCV response curve (shape: 256, 3) for RGB channels
    """
    print("\n=== OpenCV Debevec-Malik Implementation ===")
    
    # OpenCV expects images as uint8
    images_uint8 = images.astype(np.uint8)
    
    # Create calibrate object
    calibrate = cv2.createCalibrateDebevec()
    
    # Estimate response curve
    # OpenCV expects float32 (CV_32FC1), not float64
    response = calibrate.process(images_uint8, exposure_times.astype(np.float32))
    
    print(f"OpenCV response shape: {response.shape}")
    
    # OpenCV response can be (256, 3) or (256, 1, 3) - handle both
    if len(response.shape) == 3:
        # Shape is (256, 1, 3) - squeeze middle dimension
        response = response.squeeze(1)  # Now (256, 3)
    
    # Average across RGB channels for comparison plot
    response_avg = np.mean(response, axis=1)  # (256,)
    
    # Ensure it's 1D
    response_avg = np.asarray(response_avg).flatten()
    
    # Convert to log space for comparison: g(z) = log(response(z))
    g_opencv = np.log(response_avg + 1e-10)
    
    # Ensure g_opencv is 1D
    g_opencv = np.asarray(g_opencv).flatten()
    
    # Normalize
    g_opencv = g_opencv - g_opencv[128]
    
    return response, g_opencv, response_avg


def linearize_images(images, f_inv):
    """
    Linearize JPG images using inverse response curve.
    
    For each pixel: E_linear = f_inv(Z_jpg) / t_j
    
    Args:
        images: (num_images, H, W, 3) JPG images
        f_inv: (256,) or (256, 3) lookup table mapping pixel value -> linear irradiance
               If (256,), same curve for all channels. If (256, 3), per-channel curves.
    
    Returns:
        linearized: (num_images, H, W, 3) linearized images
    """
    num_images, H, W, C = images.shape
    linearized = np.zeros_like(images, dtype=np.float32)
    
    # Handle both single curve and per-channel curves
    if len(f_inv.shape) == 1:
        # Single curve for all channels
        f_inv_per_channel = [f_inv] * C
    elif f_inv.shape[1] == C:
        # Per-channel curves (256, 3) - extract each channel
        f_inv_per_channel = [f_inv[:, c] for c in range(C)]
    else:
        # Unexpected shape, use first channel or average
        print(f"Warning: unexpected f_inv shape {f_inv.shape}, using first channel")
        f_inv_per_channel = [f_inv[:, 0]] * C
    
    for img_idx in range(num_images):
        img = images[img_idx].astype(np.int32)  # (H, W, 3)
        
        # Clip to valid range
        img = np.clip(img, 0, 255)
        
        # Apply inverse response curve for each channel
        for c in range(C):
            # Use lookup table: f_inv[z] gives linear value
            # f_inv_per_channel[c] is (256,), img[:, :, c] is (H, W)
            # Result is (H, W)
            linearized[img_idx, :, :, c] = f_inv_per_channel[c][img[:, :, c]]
        
        # Divide by exposure time to get irradiance
        # Note: exposure_times are relative, so result is up to scale
        linearized[img_idx] = linearized[img_idx] / (exposure_times[img_idx] + 1e-10)
    
    return linearized


def combine_hdr_linearized(linearized_images, exposure_times):
    """
    Combine linearized images into HDR using the same algorithm as exercise 6.
    
    Start with longest exposure, replace saturated pixels with shorter exposures.
    Important: update threshold dynamically (learned from fixing exercise 6!)
    """
    print("\n=== Combining HDR ===")
    
    num_images, H, W, C = linearized_images.shape
    
    # Start with longest exposure (index 0, t=1.0)
    hdr = linearized_images[0].copy()  # (H, W, 3)
    t = 0.8 * hdr.max()
    
    print(f"Base image (longest exposure), max={hdr.max():.1f}, threshold={t:.1f}")
    
    # Process shorter exposures
    for idx in range(1, num_images):
        # Update threshold based on current HDR state
        t = 0.8 * hdr.max()
        
        # Scale factor: same as prob6 - exposures are halved each time
        # factor = 2^idx (matches prob6 exactly)
        factor = 2**idx
        i_scaled = linearized_images[idx] * factor
        
        # Process each channel separately (like prob6 does for single channel)
        # This matches the exact logic from exercise 6
        for c in range(C):
            # Find saturated pixels in this channel
            mask = hdr[:, :, c] > t  # (H, W) - True where channel c is saturated
            
            # Replace saturated pixels in this channel
            hdr[:, :, c][mask] = i_scaled[:, :, c][mask]
        
        # Count total replaced pixels (any channel) for reporting
        mask_any = (hdr > t).any(axis=2)
        print(f"Added image {idx}, factor={factor:.2f}, replaced={mask_any.sum()} pixels, t={t:.1f}")
    
    return hdr


def apply_gray_world_white_balance(rgb):
    """Gray world white balance (same as exercise 4)."""
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    
    mi = np.mean(rgb)
    mR = np.mean(R)
    mG = np.mean(G)
    mB = np.mean(B)
    
    sR = mi / (mR + 1e-10)
    sG = mi / (mG + 1e-10)
    sB = mi / (mB + 1e-10)
    
    R_new = R * sR
    G_new = G * sG
    B_new = B * sB
    
    return np.stack([R_new, G_new, B_new], axis=-1)


def tone_map_log(rgb_hdr):
    """Log compression tone mapping (same as exercise 6)."""
    eps = 1e-8
    log_img = np.log(rgb_hdr + eps)
    
    log_min = log_img.min()
    log_max = log_img.max()
    
    log_norm = (log_img - log_min) / (log_max - log_min + 1e-8)
    log_norm = np.clip(log_norm, 0., 1.)
    
    return (log_norm * 255).astype(np.uint8)


def icam06(rgb, output_range=4.5, d=9, sigma_color=0.35, sigma_space=25.0):
    """
    iCAM06 tone mapping (from exercise 7).
    Better quality than simple log compression.
    """
    rgb = rgb.astype(np.float32)
    
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    
    # intensity as in the slide: I = (20R + 40G + B) / 61
    I = (20.0 * R + 40.0 * G + B) / 61.0
    eps = 1e-8
    I = np.maximum(I, eps)
    
    r = R / I
    g = G / I
    b = B / I
    
    log_I = np.log(I).astype(np.float32)
    
    # bilateral filter on log(I)
    log_base = cv2.bilateralFilter(log_I, d, sigma_color, sigma_space)
    log_detail = log_I - log_base
    
    comp = np.log(output_range) / (log_base.max() - log_base.min() + 1e-10)
    log_off = -log_base.max() * comp
    
    log_out = log_base * comp + log_off + log_detail
    I_out = np.exp(log_out)
    
    out = np.empty_like(rgb)
    out[..., 0] = r * I_out
    out[..., 1] = g * I_out
    out[..., 2] = b * I_out
    
    return out


def tone_map_icam06(rgb_hdr, output_range=5, d=9, sigma_color=0.35, sigma_space=20.0):
    """
    iCAM06 tone mapping wrapper (same parameters as exercise 7).
    Applies iCAM06 and then normalizes to 0-255.
    """
    icam_img = icam06(rgb_hdr,
                      output_range=output_range,
                      d=d,
                      sigma_color=sigma_color,
                      sigma_space=sigma_space)
    
    icam_img = np.maximum(icam_img, 0)
    
    # Normalize using percentiles (like exercise 7)
    low = np.percentile(icam_img, 0.5)
    high = np.percentile(icam_img, 95)
    
    icam_norm = (icam_img - low) / (high - low + 1e-10)
    icam_norm = np.clip(icam_norm, 0, 1)
    
    return (icam_norm * 255).astype(np.uint8)


def save_hdr_for_visualization(hdr_linear, filename, method='log'):
    """
    Save HDR image (linear, high dynamic range) for visualization.
    Applies basic tone mapping to convert to 8-bit.
    
    Args:
        hdr_linear: (H, W, 3) float32 HDR image in linear space
        filename: output filename
        method: 'log' for log compression
    """
    # Simple log compression for visualization
    eps = 1e-8
    log_img = np.log(hdr_linear + eps)
    log_min = log_img.min()
    log_max = log_img.max()
    log_norm = (log_img - log_min) / (log_max - log_min + 1e-8)
    img_8bit = (np.clip(log_norm, 0, 1) * 255).astype(np.uint8)
    
    imageio.imwrite(filename, img_8bit)
    print(f"  Saved intermediate: {filename}")


def plot_response_curves(g_custom, g_opencv, save_path="exercise9_response_curves.png"):
    """Plot and compare response curves."""
    z_values = np.arange(256)
    
    # Ensure both curves are 1D arrays
    g_custom = np.asarray(g_custom).flatten()
    g_opencv = np.asarray(g_opencv).flatten()
    
    # Verify shapes
    if len(g_custom) != 256:
        raise ValueError(f"g_custom must have length 256, got {len(g_custom)}")
    if len(g_opencv) != 256:
        raise ValueError(f"g_opencv must have length 256, got {len(g_opencv)}")
    
    plt.figure(figsize=(12, 5))
    
    # Plot log response curves
    plt.subplot(1, 2, 1)
    plt.plot(z_values, g_custom, 'b-', label='Custom Implementation', linewidth=2)
    plt.plot(z_values, g_opencv, 'r--', label='OpenCV Implementation', linewidth=2)
    plt.xlabel('Pixel Value z')
    plt.ylabel('g(z) = log(f^(-1)(z))')
    plt.title('Log Inverse Response Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot inverse response curves (linear space)
    plt.subplot(1, 2, 2)
    f_inv_custom = np.exp(g_custom)
    f_inv_opencv = np.exp(g_opencv)
    
    # Normalize for comparison
    f_inv_custom = f_inv_custom / (f_inv_custom[128] + 1e-10)
    f_inv_opencv = f_inv_opencv / (f_inv_opencv[128] + 1e-10)
    
    plt.plot(z_values, f_inv_custom, 'b-', label='Custom', linewidth=2)
    plt.plot(z_values, f_inv_opencv, 'r--', label='OpenCV', linewidth=2)
    plt.xlabel('Pixel Value z')
    plt.ylabel('f^(-1)(z) (Linear Irradiance)')
    plt.title('Inverse Response Curves (Linear Space)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved response curve plot: {save_path}")

def save_hdr_for_visualization(hdr_linear, filename, method='log'):
    """
    Save HDR image (linear, high dynamic range) for visualization.
    Applies basic tone mapping to convert to 8-bit.
    
    Args:
        hdr_linear: (H, W, 3) float32 HDR image in linear space
        filename: output filename
        method: 'log' for log compression, 'percentile' for percentile normalization
    """
    if method == 'log':
        # Simple log compression for visualization
        eps = 1e-8
        log_img = np.log(hdr_linear + eps)
        log_min = log_img.min()
        log_max = log_img.max()
        log_norm = (log_img - log_min) / (log_max - log_min + 1e-8)
        img_8bit = (np.clip(log_norm, 0, 1) * 255).astype(np.uint8)
    else:  # percentile
        # Percentile normalization
        low = np.percentile(hdr_linear, 1.0)
        high = np.percentile(hdr_linear, 99.0)
        norm = (hdr_linear - low) / (high - low + 1e-10)
        img_8bit = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    
    imageio.imwrite(filename, img_8bit)
    print(f"  Saved intermediate: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Exercise 9: HDR from JPG images')
    parser.add_argument('--method', type=int, choices=[1, 2], default=1,
                        help='1: Custom implementation, 2: OpenCV implementation')
    
    args = parser.parse_args()
    
    if args.method == 1:
        print("\n" + "="*60)
        print("OPTION 1: Custom Debevec-Malik Implementation")
        print("="*60)
        
        # Estimate response curve
        g_custom, f_inv_custom = estimate_response_curve_custom(images, exposure_times)
        
        # Linearize images
        print("\nLinearizing images...")
        linearized = linearize_images(images, f_inv_custom)
        
        # Combine HDR
        hdr = combine_hdr_linearized(linearized, exposure_times)
        save_hdr_for_visualization(hdr, "exercise9_step1_after_hdr.png", method='log')


        # White balance
        print("\nApplying white balance...")
        hdr_wb = apply_gray_world_white_balance(hdr)
        save_hdr_for_visualization(hdr_wb, "exercise9_step2_after_wb.png", method='log')

        # Tone mapping - using iCAM06 (better than log compression)
        print("Applying iCAM06 tone mapping...")
        
        hdr_final = tone_map_icam06(hdr_wb)
        
        # Save
        output_path = "exercise9_hdr_custom.png"
        imageio.imwrite(output_path, hdr_final)
        print(f"\nSaved HDR image: {output_path}")
        
        # Plot response curve
        # For comparison, also compute OpenCV curve
        print("\nComputing OpenCV curve for comparison...")
        response_opencv, g_opencv, _ = estimate_response_curve_opencv(images, exposure_times)
        plot_response_curves(g_custom, g_opencv, "exercise9_response_curves_custom.png")
        
    else:  # method == 2
        print("\n" + "="*60)
        print("OPTION 2: OpenCV Implementation")
        print("="*60)
        
        # Estimate response curve
        response_opencv, g_opencv, f_inv_opencv = estimate_response_curve_opencv(images, exposure_times)
        
        # Linearize images using OpenCV response
        print("\nLinearizing images...")
        # OpenCV response is per-channel (256, 3), use it directly
        linearized = linearize_images(images, response_opencv)
        
        # Combine HDR
        hdr = combine_hdr_linearized(linearized, exposure_times)
        save_hdr_for_visualization(hdr, "exercise9_opencv_step1_after_hdr.png")
        
        # White balance
        print("\nApplying white balance...")
        hdr_wb = apply_gray_world_white_balance(hdr)
        save_hdr_for_visualization(hdr_wb, "exercise9_opencv_step2_after_wb.png")
        
        # Tone mapping - using iCAM06 (better than log compression)
        print("Applying iCAM06 tone mapping...")
        hdr_final = tone_map_icam06(hdr_wb)
        
        # Save
        output_path = "exercise9_opencv_step3_after_icam.png"
        imageio.imwrite(output_path, hdr_final)
        print(f"\nSaved HDR image: {output_path}")
        
    
    print("\nDone!")


if __name__ == "__main__":
    main()

