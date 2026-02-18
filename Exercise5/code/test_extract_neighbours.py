"""
Test script for extract_neighbours function
"""
import numpy as np
from selective_search import extract_neighbours

print("=" * 60)
print("TEST: extract_neighbours")
print("=" * 60)

# Create test regions with known bounding boxes
print("\n1. Creating test regions...")
test_regions = {
    0: {
        "min_x": 10, "max_x": 30,
        "min_y": 10, "max_y": 30,
        "size": 400,
        "colour_hist": np.random.rand(75),
        "texture_hist": np.random.rand(30)
    },
    1: {
        "min_x": 25, "max_x": 45,  # Overlaps with region 0
        "min_y": 25, "max_y": 45,  # Overlaps with region 0
        "size": 400,
        "colour_hist": np.random.rand(75),
        "texture_hist": np.random.rand(30)
    },
    2: {
        "min_x": 50, "max_x": 70,  # No overlap with 0 or 1
        "min_y": 50, "max_y": 70,
        "size": 400,
        "colour_hist": np.random.rand(75),
        "texture_hist": np.random.rand(30)
    },
    3: {
        "min_x": 65, "max_x": 85,  # Overlaps with region 2
        "min_y": 65, "max_y": 85,  # Overlaps with region 2
        "size": 400,
        "colour_hist": np.random.rand(75),
        "texture_hist": np.random.rand(30)
    },
    4: {
        "min_x": 100, "max_x": 120,  # Isolated, no neighbors
        "min_y": 100, "max_y": 120,
        "size": 400,
        "colour_hist": np.random.rand(75),
        "texture_hist": np.random.rand(30)
    }
}

print(f"✓ Created {len(test_regions)} test regions")
print("  Region 0: (10,10) to (30,30)")
print("  Region 1: (25,25) to (45,45) - should neighbor 0")
print("  Region 2: (50,50) to (70,70)")
print("  Region 3: (65,65) to (85,85) - should neighbor 2")
print("  Region 4: (100,100) to (120,120) - isolated")

# Test extract_neighbours
print("\n2. Testing extract_neighbours...")
try:
    neighbours = extract_neighbours(test_regions)
    print(f"✓ extract_neighbours successful")
    print(f"  Number of neighbor pairs found: {len(neighbours)}")
    
    # Validate format
    print("\n3. Validating output format...")
    if len(neighbours) > 0:
        sample = neighbours[0]
        print(f"  Sample neighbor pair: {sample}")
        print(f"  Type: {type(sample)}")
        print(f"  Length: {len(sample)}")
        
        if len(sample) == 2:
            (ai, ar), (bi, br) = sample
            print(f"  First element: region_id={ai}, type={type(ar)}")
            print(f"  Second element: region_id={bi}, type={type(br)}")
            
            # Check structure
            if isinstance(ar, dict) and isinstance(br, dict):
                print(f"  ✓ Both elements are dictionaries")
                required_keys = ["min_x", "max_x", "min_y", "max_y", "size"]
                for key in required_keys:
                    if key in ar and key in br:
                        print(f"    ✓ Both have '{key}'")
                    else:
                        print(f"    ✗ Missing '{key}'")
            else:
                print(f"  ✗ Elements should be dictionaries")
        else:
            print(f"  ✗ Expected tuple of length 2, got {len(sample)}")
    
    # Check expected neighbors
    print("\n4. Checking expected neighbor pairs...")
    expected_pairs = set()
    expected_pairs.add((0, 1))  # Region 0 and 1 overlap
    expected_pairs.add((2, 3))  # Region 2 and 3 overlap
    
    found_pairs = set()
    for (ai, ar), (bi, br) in neighbours:
        # Normalize pair (smaller ID first)
        pair = (min(ai, bi), max(ai, bi))
        found_pairs.add(pair)
    
    print(f"  Expected pairs: {expected_pairs}")
    print(f"  Found pairs: {found_pairs}")
    
    # Check if all expected pairs are found
    missing = expected_pairs - found_pairs
    extra = found_pairs - expected_pairs
    
    if len(missing) == 0:
        print(f"  ✓ All expected pairs found")
    else:
        print(f"  ✗ Missing pairs: {missing}")
    
    if len(extra) == 0:
        print(f"  ✓ No unexpected pairs")
    else:
        print(f"  ⚠ Unexpected pairs: {extra}")
        print(f"    (This might be okay if the intersect function finds more overlaps)")
    
    # Detailed output
    print("\n5. Detailed neighbor pairs:")
    for i, ((ai, ar), (bi, br)) in enumerate(neighbours):
        print(f"  Pair {i+1}: Region {ai} <-> Region {bi}")
        print(f"    Region {ai}: ({ar['min_x']},{ar['min_y']}) to ({ar['max_x']},{ar['max_y']})")
        print(f"    Region {bi}: ({br['min_x']},{br['min_y']}) to ({br['max_x']},{br['max_y']})")
    
except Exception as e:
    print(f"✗ extract_neighbours failed: {e}")
    import traceback
    traceback.print_exc()

# Test with empty regions
print("\n6. Testing with empty regions dictionary...")
try:
    empty_neighbours = extract_neighbours({})
    print(f"✓ Empty regions handled: {len(empty_neighbours)} neighbors found")
except Exception as e:
    print(f"✗ Failed with empty regions: {e}")

# Test with single region
print("\n7. Testing with single region...")
try:
    single_region = {0: test_regions[0]}
    single_neighbours = extract_neighbours(single_region)
    print(f"✓ Single region handled: {len(single_neighbours)} neighbors found (expected 0)")
except Exception as e:
    print(f"✗ Failed with single region: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)


