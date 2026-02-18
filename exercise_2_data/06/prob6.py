import rawpy
import numpy as np
from scipy.signal import convolve2d
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

# pattern: RGGB --> this pattern was extracted the same way as ex1
Mr[(rows % 2 == 0) & (cols % 2 == 0)] = 1   # R
Mg[(rows % 2 == 0) & (cols % 2 == 1)] = 1   # G (top-right)
Mg[(rows % 2 == 1) & (cols % 2 == 0)] = 1   # G (bottom-left)
Mb[(rows % 2 == 1) & (cols % 2 == 1)] = 1   # B

K = np.ones((3,3), dtype=np.float32)

R = reconstruct_channel(Mr, X, K)
G = reconstruct_channel(Mg, X, K)
B = reconstruct_channel(Mb, X, K)

rgb = np.stack([R, G, B], axis=-1)


# gray world -> similar to ex4

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


# log compress + map to 8bit

eps=1e-8 #avoid log(0)
log_img = np.log(rgb_wb + eps)

log_min = log_img.min()
log_max=log_img.max()

log_norm = (log_img - log_min) / (log_max - log_min + 1e-8)
log_norm = np.clip(log_norm, 0., 1.)

img_8 = (log_norm*255).astype(np.uint8)
imageio.imwrite('exercise6_hdr_log.png', img_8)

print('saved exercise6_hdr_log.png')
