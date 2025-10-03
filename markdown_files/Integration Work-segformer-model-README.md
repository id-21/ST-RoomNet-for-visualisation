<!-- Original Path: Integration Work/segformer-model/README.md -->

# Wallpaper Visualizer - Gradio API with SegFormer

🎨 **AI-Powered Wall Wallpaper Visualization Tool**

Transform your interior design workflow with this intelligent wallpaper visualization application. Upload a room image and wallpaper pattern to see how different designs would look on your walls, powered by NVIDIA's SegFormer for accurate wall detection.

## ✨ Features

- **🤖 AI-Powered Wall Detection**: Uses SegFormer (NVIDIA) for semantic segmentation of wall surfaces
- **🔍 Edge-Enhanced Detection**: Advanced edge detection with Canny and Hough line transforms for improved accuracy
- **📊 Confidence Visualization**: Side-by-side comparison of detection confidence with overlay visualizations
- **🎯 Dual Processing Modes**: Compare raw segmentation vs. edge-enhanced processing results
- **🖥️ User-Friendly Interface**: Clean Gradio web interface with organized comparison views
- **⚡ Real-Time Processing**: Fast inference with GPU acceleration support

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 4GB+ RAM (8GB+ recommended)
- Optional: CUDA-compatible GPU for faster processing

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/Wallpaper-Visualiser-Gradio-API-with-Segformer.git
   cd Wallpaper-Visualiser-Gradio-API-with-Segformer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv wallpaper_env
   source wallpaper_env/bin/activate  # On Windows: wallpaper_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the interface**
   - Open your browser to `http://127.0.0.1:7860`
   - Upload a room image and wallpaper pattern
   - Click "Generate Wallpaper Visualization"

## 📖 Usage Guide

### Input Requirements
- **Room Image**: JPG/PNG format, preferably showing clear wall surfaces
- **Wallpaper Pattern**: JPG/PNG format, will be tiled to fit wall dimensions

### Interface Sections

1. **Input Panel**: Upload your room image and wallpaper pattern
2. **Results Overview**: Original room and edge-processed results
3. **Wall Detection Comparison**: Side-by-side raw vs. edge-enhanced masks
4. **Confidence Visualization**: Heat maps showing detection confidence levels

### Best Practices
- Use well-lit room images with clear wall boundaries
- Choose wallpaper patterns that tile well
- Rooms with perpendicular walls work best for the current version

## 🔧 Technical Details

### Core Technologies
- **SegFormer**: `nvidia/segformer-b2-finetuned-ade-512-512` for semantic segmentation
- **ADE20K Dataset**: Pre-trained on 150 semantic classes (Class 0: Wall)
- **OpenCV**: Edge detection, contour processing, perspective transformation
- **Gradio**: Web interface framework

### Wall Detection Pipeline
1. **Preprocessing**: Optional edge enhancement using Canny edge detection
2. **Segmentation**: SegFormer inference for wall pixel classification
3. **Edge Analysis**: Hough line detection for wall boundary identification
4. **Mask Creation**: Binary mask generation with confidence scoring
5. **Perspective Correction**: Geometric transformation for realistic wallpaper application

### Advanced Features
- **Confidence Mapping**: Per-pixel confidence scores for segmentation quality assessment
- **Edge-Based Separation**: Foundation for separating perpendicular walls (experimental)
- **Fallback Mechanisms**: Robust error handling with graceful degradation

## 📊 Model Information

- **Model**: SegFormer-B2 (NVIDIA)
- **Training Dataset**: ADE20K (150 semantic classes)
- **Target Class**: Class 0 (Wall surfaces)
- **Input Resolution**: 512x512 (automatically resized)
- **Output**: Per-pixel segmentation with confidence scores

## 🛠️ Development

### Current Implementation Status
- ✅ Basic wallpaper visualization
- ✅ Confidence comparison (raw vs. edge-enhanced)
- ✅ Side-by-side interface layout
- ✅ Edge detection foundation
- 🔄 Individual wall separation (in development)
- 📋 Interactive wall selection (planned)

### Experimental Features
This project includes experimental edge-based wall detection aimed at separating perpendicular walls in corner views. The current implementation:
- Detects horizontal and vertical line segments
- Classifies wall orientations
- Provides foundation for future wall separation algorithms

### Contributing
Contributions are welcome! Areas for improvement:
- Enhanced wall separation algorithms
- Interactive wall selection
- Support for more complex room layouts
- Performance optimizations

## 📝 Dependencies

### Core Libraries
- `torch>=2.7.0` - PyTorch for model inference
- `transformers>=4.44.0` - HuggingFace transformers
- `gradio>=5.31.0` - Web interface
- `opencv-python>=4.11.0` - Computer vision operations
- `numpy>=2.2.6` - Numerical computing
- `pillow>=11.2.1` - Image processing

### Full Requirements
See `requirements.txt` for complete dependency list with pinned versions.

## 🏗️ Architecture

```
app.py
├── detect_wall_edges()          # Edge detection and line analysis
├── separate_wall_planes()       # Wall separation (experimental)
├── create_wall_mask()          # Binary mask generation
├── segment()                   # SegFormer inference
├── visualize_confidence_map()  # Confidence visualization
└── process()                   # Main processing pipeline
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- @Sanjay Bora for creating the foundations of this app as a Floor Visualiser which was refactored for wall visualisations
- **NVIDIA** for the SegFormer model
- **MIT** for the ADE20K dataset
- **HuggingFace** for the transformers library
- **Gradio** for the web interface framework

## 🐛 Known Issues & Limitations

- Wall separation currently works best with simple perpendicular wall layouts
- Performance depends on input image quality and lighting conditions
- GPU acceleration recommended for larger images

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---


**Built with ❤️ for interior design enthusiasts and computer vision researchers**
