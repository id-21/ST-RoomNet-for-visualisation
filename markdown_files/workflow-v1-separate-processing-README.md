<!-- Original Path: workflow-v1-separate-processing/README.md -->

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
