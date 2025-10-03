<!-- Original Path: README.md -->

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
