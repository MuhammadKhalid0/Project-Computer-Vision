import rawpy
import numpy as np
import cv2


def _unsharp(rgb, amount=0.3, blur_sigma=1.0):
    """
    Very light unsharp mask on RGB image in [0,1].
    """
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    blurred = cv2.GaussianBlur(rgb, (0, 0), blur_sigma)
    sharpened = rgb + amount * (rgb - blurred)
    return np.clip(sharpened, 0.0, 1.0)


def process_raw(input_path, output_jpg_path):
    """
    Exercise 8 pipeline aimed at *realistic* colour:
    - rawpy demosaicing + camera white balance
    - rawpy tone curve & gamma & auto-bright
    - light unsharp for crispness
    - save high-quality JPEG
    """
    with rawpy.imread(input_path) as raw:
        rgb8 = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,   # or DCB/ Adaptive Homogeneity-Directed demosaicing
            use_camera_wb=True,                              # camera WB
            no_auto_bright=False,                            # let it set brightness/ automatic brightness adjustment.
            output_bps=8,                                    # 8-bit sRGB
            output_color=rawpy.ColorSpace.sRGB               # standard colour space
            # gamma left as default (2.222, 4.5)
        )

    # rawpy already returned an 8-bit sRGB image
    rgb = rgb8.astype(np.float32) / 255.0

    # Optional: very light local sharpening, no colour shifts
    rgb_sharp = _unsharp(rgb, amount=0.3, blur_sigma=1.0)

    rgb_final = np.clip(rgb_sharp, 0.0, 1.0)
    rgb_8 = (rgb_final * 255.0 + 0.5).astype(np.uint8)

    # Save as high-quality JPEG
    cv2.imwrite(
        output_jpg_path,
        cv2.cvtColor(rgb_8, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 99]
    )


if __name__ == "__main__":
    process_raw("IMG_4782.CR3", "IMG_4782_realcolor.jpg")
    print("Saved IMG_4782_realcolor.jpg")
