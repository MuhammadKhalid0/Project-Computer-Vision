# Group Assignment 1: "Box Dimension Estimation from Point Cloud Data"

## description: 
  This assignment estimates the height, width, and length of a box using 3D point cloud
  and amplitude image data. The method uses RANSAC plane fitting to detect the floor
  and box-top planes, then computes dimensions from their geometric relationship.
  For full explanation, implementation details, and results — please refer to the
  report (PDF) included in this repository.

## requirements:
  - numpy
  - scipy
  - matplotlib
  - scikit-image

## usage: 
  Run the main script from the command line:
    python starter.py <mat_path> [--example N] [--th_floor T1] [--th_top T2] [--max-itr M] [--save-viz]

  Example:
    python RANSAC_Box_Dimension_Estimation.py Examples/example1kinect.mat --example 1 --th_floor 0.01 --th_top 0.01 --save-viz

## arguments:
  mat_path: "Path to the .mat file containing amplitude, distance, and point cloud data"
  --example: "Example number (default: 1)"
  --th_floor: "RANSAC threshold for floor plane (default: 0.01)"
  --th_top: "RANSAC threshold for box-top plane (default: 0.01)"
  --max-itr: "Number of RANSAC iterations (default: 10000)"
  --save-viz: "Save visualization images (floor/box-top masks, overlays, etc.)"

## notes: 
  - Outputs (images and masks) are saved in a Results/ folder when --save-viz is used.
  - The estimated box height, length, and width are printed in the console.
  - More details on methodology and results are provided in the report PDF.
