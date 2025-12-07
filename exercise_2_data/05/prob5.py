import numpy as np
import rawpy
import matplotlib.pyplot as plt

files = ["IMG_3044.CR3","IMG_3045.CR3","IMG_3046.CR3",
         "IMG_3047.CR3","IMG_3048.CR3","IMG_3049.CR3"]
exposure_times = np.array([1/10, 1/20, 1/40, 1/80, 1/160, 1/320], dtype=np.float32)

avg_values = []

for f in files:
    with rawpy.imread(f) as raw:
        raw_arr = np.array(raw.raw_image_visible, dtype=np.float32)
        avg_values.append(raw_arr.mean())

avg_values = np.array(avg_values)

# estimate black level as intercept of line fit
coef = np.polyfit(exposure_times, avg_values, 1)   # slope, intercept
slope, black = coef
print("Estimated black level:", black)

signal_values = avg_values - black

print("Ratios signal[i] / signal[i+1]:",
      signal_values[:-1] / signal_values[1:])

plt.figure()
plt.plot(exposure_times, avg_values, "o-", label="avg(raw)")
plt.plot(exposure_times, slope*exposure_times + black, "--", label="linear fit")
plt.xlabel("Exposure time (s)")
plt.ylabel("Average raw value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ex5_linearity_with_fit.png", dpi=200)
