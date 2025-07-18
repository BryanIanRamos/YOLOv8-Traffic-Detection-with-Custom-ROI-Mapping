# YOLOv8 Traffic Detection with Custom ROI Mapping

A complete traffic monitoring system using YOLOv8 object detection with custom polygon-based Region of Interest (ROI) mapping for precise traffic analysis.

## 🚗 Features

- **Real-time traffic detection** using YOLOv8 (cars, trucks, buses, bicycles, persons, traffic lights)
- **Custom polygon ROI mapping** - Define exact road areas to monitor
- **Interactive ROI creator** - Click and draw your monitoring zones
- **Multi-format output** - JSON, Python code, annotated images, and text summaries
- **Error handling** - Robust video processing with comprehensive error management
- **Visual feedback** - Real-time display of detections and ROI boundaries

## 📁 Project Structure

```
Yolov8-Traffic/
├── main.py                    # Main traffic detection script
├── bounding_box_mapper.py     # Interactive ROI polygon creator
├── intersectionRoad1.mp4      # Your traffic video file
├── README.md                  # This file
└── roi_mapping_results/       # Generated ROI configurations
    └── session_YYYYMMDD_HHMMSS/
        ├── original_frame.jpg
        ├── roi_regions_annotated.jpg
        ├── roi_polygons.json
        ├── roi_code_for_main.py
        └── roi_summary.txt
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

### Download YOLOv8 Model

The YOLOv8 model will be automatically downloaded on first run.

## 🚀 Quick Start

### 1. Create Your ROI Regions

First, define the areas you want to monitor using the interactive mapper:

```bash
python bounding_box_mapper.py
```

**Controls:**

- **Left Click**: Add point to polygon
- **Right Click**: Finish current polygon (minimum 3 points)
- **'s'**: Save all polygons
- **'c'**: Clear current polygon
- **'r'**: Reset all polygons
- **'q'**: Quit and auto-save

### 2. Run Traffic Detection

```bash
python main.py
```

## 📊 Visual Example

### ROI Creation Process

![ROI Creation](https://via.placeholder.com/640x360/FF0000/FFFFFF?text=Interactive+ROI+Mapping)

### Detection Results

![Detection Results](https://via.placeholder.com/640x360/00FF00/000000?text=Traffic+Detection+Output)

## 📝 Configuration

### Detected Object Classes

The system detects these COCO classes by default:

- **Person** (ID: 1)
- **Bicycle** (ID: 2)
- **Car** (ID: 3)
- **Bus** (ID: 5)
- **Truck** (ID: 7)
- **Traffic Light** (ID: 9)

### YOLOv8 Model Variants

Choose your model based on speed vs accuracy needs:

| Model        | Size  | Speed      | Accuracy   | Use Case               |
| ------------ | ----- | ---------- | ---------- | ---------------------- |
| `yolov8n.pt` | 6MB   | ⭐⭐⭐⭐⭐ | ⭐⭐       | Real-time applications |
| `yolov8s.pt` | 22MB  | ⭐⭐⭐⭐   | ⭐⭐⭐     | Balanced performance   |
| `yolov8m.pt` | 50MB  | ⭐⭐⭐     | ⭐⭐⭐⭐   | Production systems     |
| `yolov8l.pt` | 87MB  | ⭐⭐       | ⭐⭐⭐⭐⭐ | High accuracy needed   |
| `yolov8x.pt` | 136MB | ⭐         | ⭐⭐⭐⭐⭐ | Maximum accuracy       |

## 🔧 Customization

### Change Detection Classes

Edit `allowed_class_ids` in `main.py`:

```python
# Example: Only detect vehicles
allowed_class_ids = [3, 5, 7]  # car, bus, truck
```

### Modify Video Source

Change the video path in both files:

```python
video_path = "your_video_file.mp4"
```

### Adjust Model

Switch YOLOv8 variant in `main.py`:

```python
model = YOLO('yolov8s.pt')  # Change to desired model
```

## 📤 Output Files

### Automatic Generation

When you create ROI regions, the system generates:

1. **`roi_polygons.json`** - Machine-readable polygon coordinates
2. **`roi_code_for_main.py`** - Ready-to-use Python code
3. **`roi_summary.txt`** - Human-readable summary
4. **`original_frame.jpg`** - Clean reference frame
5. **`roi_regions_annotated.jpg`** - Visual ROI overlay

### Sample Generated Code

```python
# road1 polygon points
road1_points = [(286, 89), (469, 116), (500, 51), (339, 33), (285, 89)]
road1_polygon = np.array(road1_points, np.int32)

# road2 polygon points
road2_points = [(522, 158), (452, 308), (600, 336), (632, 186), (521, 158)]
road2_polygon = np.array(road2_points, np.int32)

# Function to check if point is inside polygon
def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0
```

## 🎯 How It Works

### 1. ROI Definition

- Load video frame
- Interactive polygon creation
- Export coordinates and code

### 2. Object Detection

- YOLOv8 processes each frame
- Detects all objects in scene
- Filters by allowed classes

### 3. ROI Filtering

- Calculate object center points
- Check if center is inside ROI polygons
- Only display objects within defined areas

### 4. Visualization

- Draw bounding boxes on detected objects
- Overlay ROI polygon boundaries
- Display confidence scores and labels

## 🔍 Troubleshooting

### Common Issues

**Video file not found:**

```
Error: Video file 'intersectionRoad1.mp4' not found!
```

→ Ensure your video file exists in the project directory

**ROI extends beyond frame:**

```
Warning: ROI1 extends beyond frame boundaries!
```

→ Use the bounding box mapper to create proper ROI regions

**Model download fails:**

```
Error loading YOLOv8 model
```

→ Check internet connection for model download

**Poor detection accuracy:**
→ Try a larger YOLOv8 model (yolov8s.pt or yolov8m.pt)

### Performance Tips

- **For real-time processing**: Use `yolov8n.pt`
- **For better accuracy**: Use `yolov8s.pt` or larger
- **For low-end hardware**: Reduce video resolution
- **For better ROI precision**: Use more polygon points

## 📊 Example Results

### Detection Statistics

```
Video loaded: 640x360, 30.00 FPS, 1443 frames
Loaded polygon ROI: road1 (5 points), road2 (5 points)
Starting video processing... Press 'q' to quit

Frame: 150/1443
Detected in road1: 2 cars, 1 truck
Detected in road2: 1 car, 1 bicycle
```

### Output Visualization

- ✅ **Green boxes**: Detected objects within ROI
- 🔵 **Blue lines**: road1 polygon boundary
- 🔴 **Red lines**: road2 polygon boundary
- 📊 **Text overlay**: Object labels and confidence scores

## 📜 License

This project is open source. Feel free to modify and distribute.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions:

- Check the troubleshooting section
- Review error messages for specific guidance
- Ensure all dependencies are properly installed

---

**Built with ❤️ using YOLOv8 and OpenCV**
