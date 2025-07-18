# Title: YOLOv8 Object Detection with Region and Class Filtering on Video

from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys

# --- CONFIGURATION ---

# COCO class IDs to keep (e.g., 1=person, 2=bicycle, 3=car, 5=bus, 7=truck, 9=traffic light)
allowed_class_ids = [1, 2, 3, 5, 7, 9]

# Define 2 custom regions (bounding boxes): (x1, y1), (x2, y2)
# These will be automatically calculated based on video dimensions
roi1 = None  # Will be set automatically
roi2 = None  # Will be set automatically

# Path to video file (downloaded video)
video_path = "intersectionRoad1.mp4"

# --- ERROR HANDLING & VALIDATION ---

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found!")
    print("Please check the file path and make sure the video exists.")
    sys.exit(1)

# Check if file is readable
if not os.access(video_path, os.R_OK):
    print(f"Error: Cannot read video file '{video_path}' - permission denied!")
    sys.exit(1)

# --- LOAD MODEL ---
try:
    model = YOLO('yolov8n.pt')  # Automatically downloads if not found
    print("YOLOv8 model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    sys.exit(1)

# --- VIDEO CAPTURE ---
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    print("Possible causes:")
    print("- File is corrupted")
    print("- Unsupported video format")
    print("- File is being used by another application")
    sys.exit(1)

# Get video info for resizing window
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video loaded: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")

# ========================================
# POLYGON ROI CONFIGURATION
# ========================================
# road1 polygon points
road1_points = [(286, 89), (469, 116), (500, 51), (339, 33), (285, 89)]
road1_polygon = np.array(road1_points, np.int32)

# road2 polygon points
road2_points = [(522, 158), (452, 308), (600, 336), (632, 186), (521, 158)]
road2_polygon = np.array(road2_points, np.int32)

# Function to check if point is inside polygon
def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

print(f"Loaded polygon ROI: road1 ({len(road1_points)} points), road2 ({len(road2_points)} points)")

# ========================================
# END OF POLYGON ROI CONFIGURATION
# ========================================

def is_inside_roi(xyxy, roi):
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    roi_x1, roi_y1 = roi[0]
    roi_x2, roi_y2 = roi[1]
    return (x1 >= roi_x1 and y1 >= roi_y1 and x2 <= roi_x2 and y2 <= roi_y2)

print("Starting video processing... Press 'q' to quit")

try:
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        frame_count += 1
        
        try:
            results = model(frame)[0]
        except Exception as e:
            print(f"Error during inference on frame {frame_count}: {e}")
            continue

        for det in results.boxes:
            try:
                cls_id = int(det.cls.item())
                if cls_id not in allowed_class_ids:
                    continue

                xyxy = det.xyxy[0].tolist()
                conf = float(det.conf.item())

                # Get center point of detected object
                center_x = int((xyxy[0] + xyxy[2]) / 2)
                center_y = int((xyxy[1] + xyxy[3]) / 2)

                # Filter only if the center point is inside either polygon ROI
                if not (is_inside_polygon(center_x, center_y, road1_polygon) or 
                        is_inside_polygon(center_x, center_y, road2_polygon)):
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                label = model.names[cls_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing detection on frame {frame_count}: {e}")
                continue

        # Draw polygon ROIs
        cv2.polylines(frame, [road1_polygon], True, (255, 0, 0), 2)
        cv2.polylines(frame, [road2_polygon], True, (0, 0, 255), 2)
        
        # Add ROI labels
        cv2.putText(frame, "road1", (road1_points[0][0], road1_points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "road2", (road2_points[0][0], road2_points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Detection with ROI + Class Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested by user")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user (Ctrl+C)")
except Exception as e:
    print(f"Unexpected error during video processing: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources cleaned up successfully")
