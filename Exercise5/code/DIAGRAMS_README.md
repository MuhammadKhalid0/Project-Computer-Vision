# Pipeline Diagrams for Report

Three Mermaid diagrams have been created for the balloon detection pipeline:

1. **`diagram_training.mmd`** - Training Phase (Tasks 5.2.1-5.2.3)
2. **`diagram_inference.mmd`** - Inference Phase (Task 5.2.4)
3. **`diagram_evaluation.mmd`** - Evaluation Phase (Task 5.2.5)

## How to Use

### Option 1: Online Mermaid Editor (Recommended)
1. Go to **https://mermaid.live**
2. Copy the contents of one of the `.mmd` files
3. Paste into the editor
4. Export as PNG/SVG for your report

### Option 2: View All Three in Browser
1. Open `view_diagrams.html` in your web browser
2. All three diagrams will render automatically
3. Take screenshots or use browser print-to-PDF

### Option 3: Use in Markdown/LaTeX
Many tools support Mermaid natively:
- **GitHub/GitLab** markdown files
- **Obsidian**, **Notion**, **Confluence**
- **LaTeX** with `mermaid` package
- **VS Code** with Mermaid extension

## File Locations

- `diagram_training.mmd` - Training pipeline
- `diagram_inference.mmd` - Inference pipeline  
- `diagram_evaluation.mmd` - Evaluation pipeline
- `view_diagrams.html` - All three in one HTML page

## Color Scheme

- **Blue boxes** = Training scripts (Tasks 5.2.1-5.2.3)
- **Green boxes** = Inference script (Task 5.2.4)
- **Purple boxes** = Evaluation script (Task 5.2.5)
- **Orange boxes** = Tuning/auxiliary scripts
- **Beige boxes** = Data files (JSON, NPZ, joblib)
- **Gray boxes** = Input data








