# File: exercise4/src/cvproj_exc/analyze_clustering.py
import numpy as np
import matplotlib.pyplot as plt
from cvproj_exc.face_recognition import FaceClustering
from cvproj_exc.face_detector import FaceDetector
import cv2
from pathlib import Path
from cvproj_exc.config import Config

def analyze_convergence(num_runs=10, num_clusters=2):
    """
    Analyze k-means convergence behavior and initialization sensitivity.
    
    Steps:
    1. Collect embeddings from training data
    2. Run k-means multiple times with different random seeds
    3. Plot convergence curves
    4. Analyze sensitivity to initialization
    """
    
    # Step 1: Collect embeddings (or load existing)
    # Create a temporary instance just to load embeddings
    temp_clustering = FaceClustering(num_clusters=2)  # Dummy k value, we'll override
    
    # Option A: Load existing embeddings from gallery
    if Path(Config.CLUSTER_GALLERY).exists():
        temp_clustering.load()
        embeddings = temp_clustering.embeddings.copy()
        print(f"Loaded {len(embeddings)} embeddings from gallery")
    else:
        # Option B: Collect new embeddings
        print("Collecting embeddings...")
        detector = FaceDetector()
        # Add your data collection code here
        # For each face: temp_clustering.partial_fit(face.aligned)
        embeddings = temp_clustering.embeddings.copy()
    
    if len(embeddings) == 0:
        print("ERROR: No embeddings available. Run training first.")
        return
    
    # Step 2: Run k-means multiple times with different seeds
    convergence_data = []
    for seed in range(num_runs):
        print(f"Running k-means with seed {seed}...")
        np.random.seed(seed)
        
        # Create new clustering instance WITHOUT loading saved state
        # We'll manually initialize to avoid auto-loading from gallery
        test_clustering = FaceClustering.__new__(FaceClustering)  # Create without calling __init__
        # Manually initialize only what we need (without loading saved state)
        from cvproj_exc.face_recognition import FaceNet
        test_clustering.facenet = FaceNet()  # Create new FaceNet instance
        test_clustering.embeddings = embeddings.copy()  # Use fresh copy of embeddings
        test_clustering.num_clusters = num_clusters
        test_clustering.max_iter = 200
        # Initialize cluster_center as empty array (size 0) to force random initialization in fit()
        # Don't use np.empty() as it creates uninitialized values that pass the size check
        test_clustering.cluster_center = np.array([]).reshape(0, embeddings.shape[1])
        test_clustering.cluster_membership = []
        
        objective_history = test_clustering.fit()
        
        if objective_history:
            convergence_data.append({
                'seed': seed,
                'iterations': len(objective_history),
                'final_objective': objective_history[-1],
                'initial_objective': objective_history[0],
                'history': objective_history
            })
            print(f"  Converged in {len(objective_history)} iterations")
            print(f"  Final objective: {objective_history[-1]:.2f}")
    
    # Step 3: Plot convergence curves
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Convergence curves
    plt.subplot(1, 2, 1)
    for data in convergence_data:
        iters = np.arange(1, len(data['history']) + 1)
        plt.plot(iters, data['history'], label=f"Seed {data['seed']}", alpha=0.7, marker='o') #plot the objective function history for each seed
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function J')
    plt.title('k-Means Convergence Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Final objective values
    plt.subplot(1, 2, 2)
    seeds = [d['seed'] for d in convergence_data]
    final_objectives = [d['final_objective'] for d in convergence_data]
    plt.bar(seeds, final_objectives)
    plt.xlabel('Random Seed')
    plt.ylabel('Final Objective Value')
    plt.title('Sensitivity to Initialization')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_filename = f'../data/convergence_analysis_k{num_clusters}.png'
    plt.savefig(plot_filename, dpi=150)
    plt.close()  # Close figure to free memory
    print(f"\nConvergence plot saved to: {plot_filename}")
    
    # Step 4: Print statistics
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS RESULTS")
    print("="*60)
    print(f"Number of runs: {num_runs}")
    print(f"Number of clusters: {num_clusters}")
    print(f"Number of samples: {len(embeddings)}")
    print("\nIteration Statistics:")
    iterations = [d['iterations'] for d in convergence_data]
    print(f"  Mean iterations: {np.mean(iterations):.2f}")
    print(f"  Std iterations: {np.std(iterations):.2f}")
    print(f"  Min iterations: {np.min(iterations)}")
    print(f"  Max iterations: {np.max(iterations)}")
    
    print("\nFinal Objective Statistics:")
    print(f"  Mean final objective: {np.mean(final_objectives):.2f}")
    print(f"  Std final objective: {np.std(final_objectives):.2f}")
    print(f"  Min final objective: {np.min(final_objectives):.2f}")
    print(f"  Max final objective: {np.max(final_objectives):.2f}")
    print(f"  Range: {np.max(final_objectives) - np.min(final_objectives):.2f}")
    
    # Sensitivity analysis
    relative_variation = (np.std(final_objectives) / np.mean(final_objectives)) * 100
    print(f"\nInitialization Sensitivity:")
    print(f"  Relative variation: {relative_variation:.2f}%")
    if relative_variation < 5:
        print("  → Low sensitivity: Algorithm finds similar solutions")
    elif relative_variation < 15:
        print("  → Moderate sensitivity: Some variation in solutions")
    else:
        print("  → High sensitivity: Solutions vary significantly with initialization")
    
    # Create convergence table
    print("\n" + "="*60)
    print("CONVERGENCE TABLE")
    print("="*60)
    print(f"{'Seed':<8} {'Iterations':<12} {'Initial J':<15} {'Final J':<15} {'Reduction':<15}")
    print("-"*60)
    for data in convergence_data:
        reduction = ((data['initial_objective'] - data['final_objective']) / data['initial_objective']) * 100
        print(f"{data['seed']:<8} {data['iterations']:<12} {data['initial_objective']:<15.2f} {data['final_objective']:<15.2f} {reduction:<15.2f}%")
    
    return convergence_data

if __name__ == "__main__":
    # Analyze with different k values
    for k in [2, 3, 4, 5, 6]:
        print(f"\n{'='*60}")
        print(f"ANALYZING WITH k={k}")
        print(f"{'='*60}")
        analyze_convergence(num_runs=10, num_clusters=k)
        print("\n")