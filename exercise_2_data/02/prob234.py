import numpy as np
import rawpy
from scipy.signal import convolve2d
import imageio.v2 as imageio

## Problem 2w
def reconstruct_channel(Mc, X, K):
    num = convolve2d(Mc * X, K, mode='same', boundary='symm')
    den = convolve2d(Mc,      K, mode='same', boundary='symm')
    den[den == 0] = 1e-8      # avoid division by zero
    return num / den


raw = rawpy.imread("IMG_4782.CR3")
X = np.array(raw.raw_image_visible)
X = X.astype(np.float32)

# Using the same Bayer filter structure as prob1: 
# G B
# R G

H, W = X.shape
rows, cols = np.indices((H, W))

Mr = np.zeros_like(X, dtype=np.float32)
Mg = np.zeros_like(X, dtype=np.float32)
Mb = np.zeros_like(X, dtype=np.float32)

#GBRG pattern:
#row%2 == 0,col%2 == 0 -> G
#row%2 == 0,col%2 == 1 -> B
#row%2 == 1,col%2 == 0 -> R
#row%2 == 1,col%2 == 1 -> G

Mg[(rows % 2 == 0) & (cols % 2 == 0)] = 1  # G
Mb[(rows % 2 == 0) & (cols % 2 == 1)] = 1  # B
Mr[(rows % 2 == 1) & (cols % 2 == 0)] = 1  # R
Mg[(rows % 2 == 1) & (cols % 2 == 1)] = 1  # G

# Kernal K, all ones as in lecture

K = np.ones((3, 3), dtype=np.float32)

R = reconstruct_channel(Mr, X, K)
G = reconstruct_channel(Mg, X, K)
B = reconstruct_channel(Mb, X, K)

rgb = np.stack([R, G, B], axis=-1)   # shape: (H, W, 3), float32

# normalize to [0,1] for viewing/export
rgb_norm = rgb / rgb.max()

# convert to 8-bit
rgb_8 = np.clip(rgb_norm * 255, 0, 255).astype(np.uint8)

imageio.imwrite('IMG_4782_demosaiced_prob2.png', rgb_8)


## Problem 3
def normalize_with_percentiles(data, p_low=0.01, p_high=99.99):
    """
    Returns normalized data + the  (a, b) values so we can invert later.
    """
    a = np.percentile(data,p_low)
    b = np.percentile(data,p_high)

    data_norm = (data-a)/(b-a)
    data_norm[data_norm<0]=0
    data_norm[data_norm>1] = 1

    return data_norm,a,b

def denormalize_from_percentiles(data_norm, a, b):
    """
    Invert the normalization: map [0,1] back to the original range approx.
    """
    return data_norm * (b - a) + a

def apply_gamma(data, gamma=0.3):
    data_norm, a, b = normalize_with_percentiles(data)
    data_gamma = np.power(data_norm, gamma)     # y = x^gamma
    return denormalize_from_percentiles(data_gamma, a, b)

def apply_sqrt_curve(data): # Another brightness curve than gamma
    data_norm, a, b = normalize_with_percentiles(data)
    data_sqrt = np.sqrt(data_norm)             #y = sqrt(x)
    return denormalize_from_percentiles(data_sqrt, a, b)

def apply_sigmoid_curve(data, s=6.0):
    data_norm, a, b = normalize_with_percentiles(data)
    y = 1 / (1 + np.exp(-s * (data_norm - 0.5)))
    return denormalize_from_percentiles(y, a, b)


gamma_rgb = np.zeros_like(rgb, dtype=np.float32)
sqrt_rgb  = np.zeros_like(rgb, dtype=np.float32)
sigmoid_rgb = np.zeros_like(rgb, dtype=np.float32)


for c in range(3):
    gamma_rgb[..., c] = apply_gamma(rgb[..., c], gamma=0.3)
    sqrt_rgb[..., c]  = apply_sqrt_curve(rgb[..., c])
    sigmoid_rgb[..., c] = apply_sigmoid_curve(rgb[..., c], s=6.0)

# Save gamma-corrected version
gamma_norm = gamma_rgb / gamma_rgb.max()
gamma_8 = np.clip(gamma_norm * 255, 0, 255).astype(np.uint8)
imageio.imwrite('IMG_4782_gamma_prob3.png', gamma_8)

# Save sqrt-curve version
sqrt_norm = sqrt_rgb / sqrt_rgb.max()
sqrt_8 = np.clip(sqrt_norm * 255, 0, 255).astype(np.uint8)
imageio.imwrite('IMG_4782_sqrt_prob3.png', sqrt_8)

# Save sigmoid-curve version
sigmoid_norm = sigmoid_rgb / sigmoid_rgb.max()
sigmoid_8 = np.clip(sigmoid_norm * 255, 0, 255).astype(np.uint8)
imageio.imwrite('IMG_4782_sigmoid_prob3.png', sigmoid_8)

# Problem 4
## Problem 4 – White Balance (Gray World)

def apply_gray_world_white_balance(rgb):
    """
    rgb: (H, W, 3) float32, linear (from demosaicing).
    Returns a white-balanced image, still in linear space.
    Gray world: multiply each channel c by mi / mc.
    """
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    # mean of entire image (all channels)
    mi = np.mean(rgb)

    # mean of each channel
    mR = np.mean(R)
    mG = np.mean(G)
    mB = np.mean(B)

    # scale factors mi / mc
    sR = mi / mR
    sG = mi / mG
    sB = mi / mB

    R_new = R * sR
    G_new = G * sG
    B_new = B * sB

    return np.stack([R_new, G_new, B_new], axis=-1)

# --- apply WB to the *linear* rgb from Problem 2 ---
rgb_wb = apply_gray_world_white_balance(rgb)

# Bring to roughly 0–255 using a high percentile (avoid hot pixels)
max_val = np.percentile(rgb_wb, 99.99)
if max_val < 1e-6:
    max_val = 1e-6  # just in case, avoid division by zero

scale = 255.0 / max_val
rgb_wb_scaled  = rgb_wb * scale
rgb_wb_clipped = np.clip(rgb_wb_scaled, 0, 255).astype(np.uint8)

imageio.imwrite('IMG_4782_wb_prob4.png', rgb_wb_clipped)