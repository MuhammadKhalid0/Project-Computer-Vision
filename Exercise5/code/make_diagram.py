"""
Generate a pipeline diagram for the balloon detection report.
Outputs: ../results/pipeline_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(ax, x, y, w, h, text, color='#4A90D9', text_color='white',
             fontsize=9, style='round,pad=0.1'):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=style,
        facecolor=color, edgecolor='#2C3E50', linewidth=1.2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold',
            linespacing=1.4)


def draw_file_box(ax, x, y, w, h, text, fontsize=8):
    """Draw a file/data box (lighter style)."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.08',
        facecolor='#F5F5DC', edgecolor='#8B8682', linewidth=1.0,
        linestyle='-')
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color='#2C3E50', family='monospace',
            linespacing=1.3)


def draw_arrow(ax, x1, y1, x2, y2, color='#2C3E50'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=1.5, connectionstyle='arc3,rad=0'))


def draw_side_arrow(ax, x1, y1, x2, y2, color='#888888'):
    """Draw a dashed side arrow (for auxiliary connections)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=1.2, linestyle='dashed',
                    connectionstyle='arc3,rad=0.2'))


def main():
    fig, ax = plt.subplots(1, figsize=(10, 16))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 17)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(0, 16.5, 'Balloon Detection Pipeline', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2C3E50')
    ax.text(0, 16.1, 'Exercise 5.2 — Selective Search + CNN Features + SVM',
            ha='center', va='center', fontsize=10, color='#666666')

    # Phase labels
    ax.text(-4.5, 15.3, 'TRAINING', fontsize=11, fontweight='bold',
            color='#E74C3C', rotation=0)
    ax.axhline(y=15.1, xmin=0.05, xmax=0.95, color='#E74C3C',
               linewidth=0.8, linestyle='-')

    ax.text(-4.5, 5.3, 'INFERENCE', fontsize=11, fontweight='bold',
            color='#27AE60', rotation=0)
    ax.axhline(y=5.1, xmin=0.05, xmax=0.95, color='#27AE60',
               linewidth=0.8, linestyle='-')

    ax.text(-4.5, 2.3, 'EVALUATION', fontsize=11, fontweight='bold',
            color='#8E44AD', rotation=0)
    ax.axhline(y=2.1, xmin=0.05, xmax=0.95, color='#8E44AD',
               linewidth=0.8, linestyle='-')

    # ---- TRAINING PHASE ----

    # Step 1: Dataset
    draw_box(ax, -2.5, 14.5, 3.0, 0.7,
             'Balloon Dataset\n(train / valid / test)',
             color='#95A5A6', fontsize=8)

    # Step 1: Generate proposals
    draw_box(ax, 0, 14.5, 3.8, 0.7,
             '1. generate_proposals.py\nSelective Search (scale=500)',
             color='#3498DB', fontsize=8)

    draw_arrow(ax, -1.0, 14.5, -1.9, 14.5)  # dataset -> generate

    # Proposals output
    draw_file_box(ax, 0, 13.4, 3.0, 0.55,
                  'proposals_train/valid/test.json')

    draw_arrow(ax, 0, 14.1, 0, 13.7)

    # Tune thresholds (side)
    draw_box(ax, -3.2, 12.4, 2.8, 0.7,
             'tune_thresholds.py\nSweep tp, tn values',
             color='#E67E22', fontsize=8)

    draw_side_arrow(ax, -1.5, 13.2, -2.5, 12.8)

    # Step 2: Create samples
    draw_box(ax, 0, 12.4, 3.8, 0.7,
             '2. create_samples.py\nIoU labeling (tp=0.4, tn=0.1)',
             color='#3498DB', fontsize=8)

    draw_arrow(ax, 0, 13.1, 0, 12.8)
    draw_side_arrow(ax, -1.8, 12.4, -1.9, 12.4)

    # Samples output
    draw_file_box(ax, 0, 11.3, 3.0, 0.55,
                  'samples_train/valid.json\n(box, label, IoU)')

    draw_arrow(ax, 0, 12.0, 0, 11.6)

    # Step 3: Extract features
    draw_box(ax, 0, 10.3, 3.8, 0.7,
             '3. extract_features.py\nResNet18 → 512-dim features',
             color='#3498DB', fontsize=8)

    draw_arrow(ax, 0, 11.0, 0, 10.7)

    # Features output
    draw_file_box(ax, 0, 9.2, 3.2, 0.55,
                  'features_train/valid.npz\n(N × 512) + labels')

    draw_arrow(ax, 0, 9.9, 0, 9.5)

    # Step 4: Train SVM
    draw_box(ax, 0, 8.2, 3.8, 0.7,
             '4. train_svm.py\nSVM (RBF, C=1, balanced)',
             color='#3498DB', fontsize=8)

    draw_arrow(ax, 0, 8.9, 0, 8.6)

    # SVM metrics (side)
    draw_file_box(ax, 3.3, 8.2, 2.6, 0.55,
                  'Val: P=1.00 R=0.61\nF1=0.76')
    draw_arrow(ax, 1.9, 8.2, 2.0, 8.2)

    # Model output
    draw_file_box(ax, 0, 7.2, 3.0, 0.55,
                  'svm_model.joblib\nsvm_scaler.joblib')

    draw_arrow(ax, 0, 7.8, 0, 7.5)

    # Vertical connector to inference
    draw_arrow(ax, 0, 6.9, 0, 5.8, color='#27AE60')

    # ---- INFERENCE PHASE ----

    # detect.py box — wider to fit content
    draw_box(ax, 0, 4.6, 4.2, 1.6,
             '5. detect.py\n\n'
             '① Selective Search → proposals\n'
             '② ResNet18 → 512-dim features\n'
             '③ SVM → balloon / background\n'
             '④ Confidence filter + NMS',
             color='#27AE60', fontsize=8, text_color='white')

    draw_arrow(ax, 0, 5.0, 0, 5.4)

    # Input image (left)
    draw_box(ax, -3.5, 4.6, 2.0, 0.6,
             'Any input\nimage',
             color='#95A5A6', fontsize=8)
    draw_arrow(ax, -2.5, 4.6, -2.1, 4.6)

    # Output (right)
    draw_file_box(ax, 3.5, 4.0, 2.2, 0.55,
                  'detection_*.png\n(visualisation)')
    draw_arrow(ax, 2.1, 4.2, 2.4, 4.1)

    # Connector to evaluation
    draw_arrow(ax, 0, 3.7, 0, 2.7, color='#8E44AD')

    # ---- EVALUATION PHASE ----

    # evaluate.py
    draw_box(ax, 0, 1.5, 4.2, 1.4,
             '6. evaluate.py\n\n'
             'MABO = 0.537 (proposal quality)\n'
             'AP@.50 = 0.451 (detection quality)\n'
             'AP@[.50:.95] = 0.070',
             color='#8E44AD', fontsize=8, text_color='white')

    draw_arrow(ax, 0, 2.0, 0, 2.2)

    # Inputs to evaluate (left)
    draw_box(ax, -3.5, 1.5, 2.0, 0.6,
             'Test images\n+ GT annotations',
             color='#95A5A6', fontsize=8)
    draw_arrow(ax, -2.5, 1.5, -2.1, 1.5)

    # Output (right)
    draw_file_box(ax, 3.5, 1.5, 2.2, 0.55,
                  'eval_detections/*.png\n(red=det, green=GT)')
    draw_arrow(ax, 2.1, 1.5, 2.4, 1.5)

    # Legend
    ax.text(0, 0.2, '── Blue: training scripts  ── Green: inference  '
            '── Purple: evaluation  ── Orange: tuning',
            ha='center', va='center', fontsize=7.5, color='#666666',
            style='italic')

    plt.tight_layout()
    out_path = '../results/pipeline_diagram.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()








