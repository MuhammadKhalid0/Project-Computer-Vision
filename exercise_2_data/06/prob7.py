import numpy as np
import rawpy
from scipy.signal import convolve2d
import cv2
import imageio.v2 as imageio

#exercise 6
files = [f"{i:02d}.CR3" for i in range(11)] 

hdr_raw = None
t = None

for idx, fname in enumerate(files):
    raw = rawpy.imread(fname)
    x = raw.raw_image_visible.astype(np.float32)
    raw.close()

    if idx == 0:
        hdr_raw = x.copy()
        t = 0.8 * hdr_raw.max()
        print(f"base = {fname}, max={hdr_raw.max():.1f}, t={t:.1f}")
    else:
        # Update threshold based on current HDR state
        t = 0.8 * hdr_raw.max()
        
        # exposures are halved each time->scale with 2^idx
        factor = 2**idx
        i_scaled = x*factor
        
        mask = hdr_raw>t
        hdr_raw[mask] = i_scaled[mask]
        
        print(f"added {fname}, factor={factor}, replaced={mask.sum()} px, t={t:.1f}")

print("HDR combine done")
X = hdr_raw


# ssame as ex2

def reconstruct_channel(Mc, X, K):
    num = convolve2d(Mc*X, K, mode='same', boundary='symm')
    den = convolve2d(Mc,   K, mode='same', boundary='symm')
    den[den == 0] = 1e-8
    return num / den


H, W = X.shape
rows, cols = np.indices((H, W))

Mr = np.zeros_like(X, dtype=np.float32)
Mg = np.zeros_like(X, dtype=np.float32)
Mb = np.zeros_like(X, dtype=np.float32)

# pattern: RGGB -? this pattern was extracted the same way as ex1
Mr[(rows % 2 == 0) & (cols % 2 == 0)] = 1   # R
Mg[(rows % 2 == 0) & (cols % 2 == 1)] = 1   # G (top-right)
Mg[(rows % 2 == 1) & (cols % 2 == 0)] = 1   # G (bottom-left)
Mb[(rows % 2 == 1) & (cols % 2 == 1)] = 1   # B

K = np.ones((3,3), dtype=np.float32)

R = reconstruct_channel(Mr, X, K)
G = reconstruct_channel(Mg, X, K)
B = reconstruct_channel(Mb, X, K)

rgb = np.stack([R, G, B], axis=-1)

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

rgb_wb = apply_gray_world_white_balance(rgb)

# ---------- iCAM06 (from the slides) ----------

def icam06(rgb, output_range=4.5, d=9, sigma_color=0.35, sigma_space=25.0):
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

    comp = np.log(output_range) / (log_base.max() - log_base.min())
    log_off = -log_base.max() * comp

    log_out = log_base * comp + log_off + log_detail
    I_out = np.exp(log_out)

    out = np.empty_like(rgb)
    out[..., 0] = r * I_out
    out[..., 1] = g * I_out
    out[..., 2] = b * I_out

    return out

# rgb_hdr is your HDR after demosaicing + white balance (float32)
icam_img = icam06(rgb_wb,
                  output_range=4,
                  d=7,
                  sigma_color=90,
                  sigma_space=25.0)

icam_img = np.maximum(icam_img, 0)

low = np.percentile(icam_img, 0.5)
high = np.percentile(icam_img, 95)

icam_norm = (icam_img - low) / (high - low)
icam_norm = np.clip(icam_norm, 0, 1)

icam_8 = (icam_norm * 255).astype(np.uint8)

imageio.imwrite("exercise7_icam06.png", icam_8)