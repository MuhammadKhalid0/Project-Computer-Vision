"""
Test script for merge_regions function
"""
import numpy as np
from selective_search import merge_regions

print("=" * 60)
print("TEST: merge_regions")
print("=" * 60)

# Create test regions
print("\n1. Creating test regions...")
r1 = {
    "min_x": 10, "max_x": 30,
    "min_y": 10, "max_y": 30,
    "size": 400,
    "colour_hist": np.ones(75, dtype=np.float32) / 75.0 * 3.0,  # Normalized to sum to 3.0
    "texture_hist": np.ones(30, dtype=np.float32) / 30.0 * 3.0   # Normalized to sum to 3.0
}

r2 = {
    "min_x": 25, "max_x": 45,
    "min_y": 25, "max_y": 45,
    "size": 600,
    "colour_hist": np.ones(75, dtype=np.float32) / 75.0 * 3.0,  # Normalized to sum to 3.0
    "texture_hist": np.ones(30, dtype=np.float32) / 30.0 * 3.0   # Normalized to sum to 3.0
}

print(f"  Region 1: bbox=({r1['min_x']},{r1['min_y']}) to ({r1['max_x']},{r1['max_y']}), size={r1['size']}")
print(f"  Region 2: bbox=({r2['min_x']},{r2['min_y']}) to ({r2['max_x']},{r2['max_y']}), size={r2['size']}")

# Test merge_regions
print("\n2. Testing merge_regions...")
try:
    merged = merge_regions(r1, r2)
    print(f"✓ merge_regions successful")
    
    # Validate structure
    print("\n3. Validating merged region structure...")
    required_keys = ["min_x", "max_x", "min_y", "max_y", "size", "colour_hist", "texture_hist"]
    for key in required_keys:
        if key in merged:
            value = merged[key]
            if isinstance(value, np.ndarray):
                print(f"  ✓ {key}: shape {value.shape}, sum={np.sum(value):.4f}")
            else:
                print(f"  ✓ {key}: {value}")
        else:
            print(f"  ✗ Missing key: {key}")
    
    # Validate bounding box
    print("\n4. Validating bounding box...")
    expected_min_x = min(r1["min_x"], r2["min_x"])
    expected_max_x = max(r1["max_x"], r2["max_x"])
    expected_min_y = min(r1["min_y"], r2["min_y"])
    expected_max_y = max(r1["max_y"], r2["max_y"])
    
    print(f"  Expected bbox: ({expected_min_x}, {expected_min_y}) to ({expected_max_x}, {expected_max_y})")
    print(f"  Actual bbox: ({merged['min_x']}, {merged['min_y']}) to ({merged['max_x']}, {merged['max_y']})")
    
    if (merged["min_x"] == expected_min_x and merged["max_x"] == expected_max_x and
        merged["min_y"] == expected_min_y and merged["max_y"] == expected_max_y):
        print(f"  ✓ Bounding box is correct")
    else:
        print(f"  ✗ Bounding box mismatch!")
    
    # Validate size
    print("\n5. Validating size...")
    expected_size = r1["size"] + r2["size"]
    print(f"  Expected size: {expected_size}")
    print(f"  Actual size: {merged['size']}")
    if merged["size"] == expected_size:
        print(f"  ✓ Size is correct")
    else:
        print(f"  ✗ Size mismatch!")
    
    # Validate histograms
    print("\n6. Validating histograms...")
    print(f"  Colour histogram shape: {merged['colour_hist'].shape} (expected (75,))")
    print(f"  Colour histogram sum: {np.sum(merged['colour_hist']):.4f} (expected ~3.0)")
    print(f"  Texture histogram shape: {merged['texture_hist'].shape} (expected (30,))")
    print(f"  Texture histogram sum: {np.sum(merged['texture_hist']):.4f} (expected ~3.0)")
    
    if merged["colour_hist"].shape == (75,):
        print(f"  ✓ Colour histogram shape correct")
    else:
        print(f"  ✗ Colour histogram shape incorrect")
    
    if merged["texture_hist"].shape == (30,):
        print(f"  ✓ Texture histogram shape correct")
    else:
        print(f"  ✗ Texture histogram shape incorrect")
    
    if np.isclose(np.sum(merged["colour_hist"]), 3.0, atol=0.01):
        print(f"  ✓ Colour histogram normalized correctly")
    else:
        print(f"  ✗ Colour histogram not normalized (sum={np.sum(merged['colour_hist']):.4f})")
    
    if np.isclose(np.sum(merged["texture_hist"]), 3.0, atol=0.01):
        print(f"  ✓ Texture histogram normalized correctly")
    else:
        print(f"  ✗ Texture histogram not normalized (sum={np.sum(merged['texture_hist']):.4f})")
    
    # Test weighted combination
    print("\n7. Testing weighted histogram combination...")
    # Create regions with different histograms
    r3 = {
        "min_x": 0, "max_x": 10,
        "min_y": 0, "max_y": 10,
        "size": 100,
        "colour_hist": np.zeros(75, dtype=np.float32),
        "texture_hist": np.zeros(30, dtype=np.float32)
    }
    r3["colour_hist"][0] = 3.0  # All weight in first bin
    r3["texture_hist"][0] = 3.0
    
    r4 = {
        "min_x": 5, "max_x": 15,
        "min_y": 5, "max_y": 15,
        "size": 200,
        "colour_hist": np.zeros(75, dtype=np.float32),
        "texture_hist": np.zeros(30, dtype=np.float32)
    }
    r4["colour_hist"][1] = 3.0  # All weight in second bin
    r4["texture_hist"][1] = 3.0
    
    merged_weighted = merge_regions(r3, r4)
    
    # Check if weighting is correct
    # r3 has size 100, r4 has size 200, total 300
    # Expected: (100 * hist3 + 200 * hist4) / 300
    # hist3[0] = 3.0, hist4[1] = 3.0
    # merged[0] should be (100 * 3.0) / 300 = 1.0
    # merged[1] should be (200 * 3.0) / 300 = 2.0
    # Then normalized: sum = 3.0, so it should stay the same
    
    expected_col_0 = (100 * 3.0) / 300.0
    expected_col_1 = (200 * 3.0) / 300.0
    
    print(f"  Region 3: size=100, colour_hist[0]=3.0")
    print(f"  Region 4: size=200, colour_hist[1]=3.0")
    print(f"  Merged colour_hist[0]: {merged_weighted['colour_hist'][0]:.4f} (expected ~{expected_col_0:.4f})")
    print(f"  Merged colour_hist[1]: {merged_weighted['colour_hist'][1]:.4f} (expected ~{expected_col_1:.4f})")
    
    if np.isclose(merged_weighted['colour_hist'][0], expected_col_0, atol=0.01):
        print(f"  ✓ Weighted combination correct for bin 0")
    else:
        print(f"  ✗ Weighted combination incorrect for bin 0")
    
    if np.isclose(merged_weighted['colour_hist'][1], expected_col_1, atol=0.01):
        print(f"  ✓ Weighted combination correct for bin 1")
    else:
        print(f"  ✗ Weighted combination incorrect for bin 1")
    
    print(f"  Merged size: {merged_weighted['size']} (expected {r3['size'] + r4['size']})")
    
except Exception as e:
    print(f"✗ merge_regions failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)


