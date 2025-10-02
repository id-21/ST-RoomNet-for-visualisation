#!/bin/bash

# Repository Restructure Script for ST-RoomNet-for-visualisation
# MacOS Compatible (Apple M4)

set -e  # Exit on error

echo "=========================================="
echo "ST-RoomNet Repository Restructure"
echo "=========================================="
echo ""

# Get the current directory (assumes script is run from repo root)
REPO_ROOT=$(pwd)
echo "Repository root: $REPO_ROOT"
echo ""

# Create backup
echo "Creating backup..."
BACKUP_DIR="${REPO_ROOT}_backup_$(date +%Y%m%d_%H%M%S)"
cp -R "$REPO_ROOT" "$BACKUP_DIR"
echo "✓ Backup created at: $BACKUP_DIR"
echo ""

# Create new directory structure
echo "Creating new directory structure..."

# Workflow V1 directories
mkdir -p workflow-v1-separate-processing/1-model-inference/outputs
mkdir -p workflow-v1-separate-processing/2-gradio-visualizer

# Workflow V2 directories
mkdir -p workflow-v2-integrated-pipeline/modules

# Shared directories
mkdir -p shared

# Keep existing directories
mkdir -p images
mkdir -p weights

echo "✓ Directory structure created"
echo ""

# Copy files for Workflow V1 (Separate Processing)
echo "Setting up Workflow V1 (Separate Processing)..."

# 1-model-inference
if [ -f "test.py" ]; then
    cp test.py workflow-v1-separate-processing/1-model-inference/run_st_roomnet.py
    echo "✓ Copied test.py → run_st_roomnet.py"
fi

if [ -f "st_roomnet_model.py" ]; then
    cp st_roomnet_model.py workflow-v1-separate-processing/1-model-inference/
    echo "✓ Copied st_roomnet_model.py"
fi

if [ -f "segformer_model.py" ]; then
    cp segformer_model.py workflow-v1-separate-processing/1-model-inference/
    echo "✓ Copied segformer_model.py"
fi

# 2-gradio-visualizer
if [ -f "gradio_visualizer.py" ]; then
    cp gradio_visualizer.py workflow-v1-separate-processing/2-gradio-visualizer/
    echo "✓ Copied gradio_visualizer.py"
fi

if [ -f "visualization_utils.py" ]; then
    cp visualization_utils.py workflow-v1-separate-processing/2-gradio-visualizer/
    echo "✓ Copied visualization_utils.py"
fi

echo ""

# Copy files for Workflow V2 (Integrated Pipeline)
echo "Setting up Workflow V2 (Integrated Pipeline)..."

# Main entry point
if [ -f "gradio_visualizer_complete.py" ]; then
    cp gradio_visualizer_complete.py workflow-v2-integrated-pipeline/main.py
    echo "✓ Copied gradio_visualizer_complete.py → main.py"
fi

# Copy all modules from Integration Work/phase3_modules/ if it exists
if [ -d "Integration Work/phase3_modules" ]; then
    cp -R "Integration Work/phase3_modules/"* workflow-v2-integrated-pipeline/modules/ 2>/dev/null || true
    echo "✓ Copied modules from Integration Work/phase3_modules/"
else
    # If phase3_modules doesn't exist, copy individual module files
    module_files=(
        "st_roomnet_model.py"
        "segformer_model.py"
        "end_to_end_pipeline.py"
        "normal_computation.py"
        "wallpaper_application.py"
        "create_walls_simple.py"
        "edge_refinement_simple.py"
        "simple_mask_combination.py"
        "room_wall_layout.py"
    )
    
    for file in "${module_files[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" workflow-v2-integrated-pipeline/modules/
            echo "✓ Copied $file to modules/"
        fi
    done
fi

echo ""

# Setup shared resources
echo "Setting up shared resources..."

if [ -f "spatial_transformer_fixed.py" ]; then
    cp spatial_transformer_fixed.py shared/
    echo "✓ Copied spatial_transformer_fixed.py"
fi

if [ -f "room_wall_layout.py" ]; then
    cp room_wall_layout.py shared/
    echo "✓ Copied room_wall_layout.py"
fi

echo ""

# Create README files for each workflow
echo "Creating README files..."

cat > workflow-v1-separate-processing/README.md << 'EOF'
# Workflow V1: Separate Processing

This workflow consists of two separate steps:

## Step 1: Model Inference
Run the models and save outputs:
```bash
cd 1-model-inference
python run_st_roomnet.py
# Outputs will be saved to outputs/
```

## Step 2: Gradio Visualizer
Upload pre-computed outputs to Gradio:
```bash
cd 2-gradio-visualizer
python gradio_visualizer.py
```

This approach allows you to:
- Run models once and reuse outputs
- Test visualization separately
- Iterate on visualization without re-running models
EOF

cat > workflow-v2-integrated-pipeline/README.md << 'EOF'
# Workflow V2: Integrated Pipeline

This workflow provides end-to-end processing in a single Gradio interface.

## Running the Application
```bash
python main.py
```

The application will:
- Accept image uploads directly in Gradio
- Run ST-RoomNet and SegFormer models
- Process results in real-time
- Display visualizations immediately

All modules are organized in the `modules/` directory for clean separation of concerns.
EOF

echo "✓ Created README files"
echo ""

# Create main README
cat > README.md << 'EOF'
# ST-RoomNet for Visualization

This repository contains two workflows for room layout visualization:

## Workflows

### Workflow V1: Separate Processing
Located in `workflow-v1-separate-processing/`
- Run models separately and save outputs
- Upload pre-computed results to Gradio visualizer
- Good for: Testing, iterating on visualization

### Workflow V2: Integrated Pipeline
Located in `workflow-v2-integrated-pipeline/`
- End-to-end processing in Gradio interface
- Real-time model inference and visualization
- Good for: Production use, demonstrations

## Directory Structure

```
ST-RoomNet-for-visualisation/
├── workflow-v1-separate-processing/
│   ├── 1-model-inference/
│   └── 2-gradio-visualizer/
├── workflow-v2-integrated-pipeline/
│   ├── main.py
│   └── modules/
├── shared/
├── images/
└── weights/
```

## Getting Started

1. Install dependencies (requirements.txt)
2. Download model weights to `weights/`
3. Choose your workflow and follow the README in that directory

## Shared Resources

Common files used by both workflows are in the `shared/` directory.
EOF

echo "✓ Created main README.md"
echo ""

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Model weights
weights/

# Outputs
outputs/
*/outputs/
*/*/outputs/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# MacOS
.DS_Store

# Backup
*_backup_*/
EOF
    echo "✓ Created .gitignore"
else
    echo "✓ .gitignore already exists"
fi

echo ""
echo "=========================================="
echo "Restructure Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "- Backup created at: $BACKUP_DIR"
echo "- Workflow V1 setup in: workflow-v1-separate-processing/"
echo "- Workflow V2 setup in: workflow-v2-integrated-pipeline/"
echo "- Shared resources in: shared/"
echo ""
echo "Next steps:"
echo "1. Review the new structure"
echo "2. Test both workflows"
echo "3. Update import paths if needed"
echo "4. Remove old files from root (optional)"
echo ""
echo "To restore from backup if needed:"
echo "  rm -rf $REPO_ROOT/*"
echo "  cp -R $BACKUP_DIR/* $REPO_ROOT/"
echo ""
