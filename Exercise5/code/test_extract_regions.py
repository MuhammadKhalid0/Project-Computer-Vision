"""
Test script for extract_regions function
"""
import numpy as np
import skimage.io
from selective_search import generate_segments, extract_regions

print("=" * 60)
print("TEST: extract_regions")
print("=" * 60)

# Load a test image
print("\n1. Loading test image...")
image_path = 'data/chrisarch/ca-annun3.jpg'  # Adjust path as needed
try:
    image_orig = skimage.io.imread(image_path)
    print(f"✓ Image loaded: shape {image_orig.shape}")
except:
    print("⚠ Image file not found, creating dummy image")
    image_orig = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"✓ Dummy image created: shape {image_orig.shape}")

# Generate segments first
print("\n2. Generating segments...")
try:
    img_with_segments = generate_segments(image_orig, scale=500, sigma=0.8, min_size=20)
    print(f"✓ Segments generated: shape {img_with_segments.shape}")
    unique_labels = np.unique(img_with_segments[:, :, 3])
    print(f"  Number of unique regions: {len(unique_labels)}")
except Exception as e:
    print(f"✗ Failed to generate segments: {e}")
    exit(1)

# Extract regions
print("\n3. Extracting regions...")
try:
    R = extract_regions(img_with_segments)
    print(f"✓ extract_regions successful")
    print(f"  Number of regions in R: {len(R)}")
    
    # Validate structure
    if len(R) > 0:
        # Check first region
        first_label = list(R.keys())[0]
        first_region = R[first_label]
        
        print(f"\n4. Validating region structure...")
        print(f"  Sample region (label {first_label}):")
        
        # Check required keys
        required_keys = ["min_x", "max_x", "min_y", "max_y", "size", 
                        "colour_hist", "texture_hist"]
        for key in required_keys:
            if key in first_region:
                value = first_region[key]
                if isinstance(value, np.ndarray):
                    print(f"    {key}: shape {value.shape}, sum={np.sum(value):.4f}")
                else:
                    print(f"    {key}: {value}")
            else:
                print(f"    ✗ Missing key: {key}")
        
        # Validate bounding box
        print(f"\n5. Validating bounding boxes...")
        valid_boxes = 0
        invalid_boxes = 0
        for label, region in R.items():
            if (region["min_x"] < region["max_x"] and 
                region["min_y"] < region["max_y"] and
                region["min_x"] >= 0 and region["min_y"] >= 0 and
                region["max_x"] <= img_with_segments.shape[1] and
                region["max_y"] <= img_with_segments.shape[0]):
                valid_boxes += 1
            else:
                invalid_boxes += 1
                print(f"  ✗ Invalid bounding box for region {label}: "
                      f"({region['min_x']}, {region['min_y']}) to "
                      f"({region['max_x']}, {region['max_y']})")
        
        print(f"  Valid bounding boxes: {valid_boxes}")
        print(f"  Invalid bounding boxes: {invalid_boxes}")
        
        # Validate histogram shapes
        print(f"\n6. Validating histograms...")
        colour_hist_ok = 0
        texture_hist_ok = 0
        for label, region in R.items():
            if region["colour_hist"].shape == (75,):
                colour_hist_ok += 1
            else:
                print(f"  ✗ Invalid colour_hist shape for region {label}: "
                      f"{region['colour_hist'].shape} (expected (75,))")
            
            if region["texture_hist"].shape == (30,):
                texture_hist_ok += 1
            else:
                print(f"  ✗ Invalid texture_hist shape for region {label}: "
                      f"{region['texture_hist'].shape} (expected (30,))")
        
        print(f"  Valid colour_hist shapes: {colour_hist_ok}/{len(R)}")
        print(f"  Valid texture_hist shapes: {texture_hist_ok}/{len(R)}")
        
        # Validate histogram normalization
        print(f"\n7. Validating histogram normalization...")
        colour_norm_ok = 0
        texture_norm_ok = 0
        for label, region in R.items():
            col_sum = np.sum(region["colour_hist"])
            tex_sum = np.sum(region["texture_hist"])
            
            if np.isclose(col_sum, 3.0, atol=0.01):  # Should be ~3.0 (3 channels)
                colour_norm_ok += 1
            else:
                print(f"  ✗ Invalid colour_hist sum for region {label}: "
                      f"{col_sum:.4f} (expected ~3.0)")
            
            if np.isclose(tex_sum, 3.0, atol=0.01):  # Should be ~3.0 (3 channels)
                texture_norm_ok += 1
            else:
                print(f"  ✗ Invalid texture_hist sum for region {label}: "
                      f"{tex_sum:.4f} (expected ~3.0)")
        
        print(f"  Normalized colour_hist: {colour_norm_ok}/{len(R)}")
        print(f"  Normalized texture_hist: {texture_norm_ok}/{len(R)}")
        
        # Check size consistency
        print(f"\n8. Validating size consistency...")
        size_ok = 0
        for label, region in R.items():
            # Calculate actual size from bounding box
            bbox_size = (region["max_x"] - region["min_x"]) * \
                       (region["max_y"] - region["min_y"])
            # Size should be <= bbox_size (regions can be non-rectangular)
            if region["size"] > 0 and region["size"] <= bbox_size:
                size_ok += 1
            else:
                print(f"  ⚠ Suspicious size for region {label}: "
                      f"size={region['size']}, bbox_size={bbox_size}")
        
        print(f"  Reasonable sizes: {size_ok}/{len(R)}")
        
    else:
        print("  ✗ No regions extracted!")
        
except Exception as e:
    print(f"✗ extract_regions failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)


