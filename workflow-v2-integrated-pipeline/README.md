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
