import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime

# Configuration
video_path = "intersectionRoad1.mp4"
output_folder = "roi_mapping_results"

# Extract video filename without extension for folder name
video_filename = os.path.splitext(os.path.basename(video_path))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_folder = os.path.join(output_folder, f"{video_filename}_{timestamp}")

# Create output folders
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created main folder: {output_folder}")

if not os.path.exists(session_folder):
    os.makedirs(session_folder)
    print(f"Created session folder: {session_folder}")

class PolygonROIMapper:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.current_polygon = []
        self.polygons = []
        self.polygon_names = []
        self.drawing = False
        self.current_name = ""
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current polygon
            self.current_polygon.append((x, y))
            print(f"Added point: ({x}, {y})")
            self.draw_current_polygon()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current polygon
            if len(self.current_polygon) >= 3:
                self.finish_polygon()
            else:
                print("Need at least 3 points for a polygon!")
                
        elif event == cv2.EVENT_MOUSEMOVE:
            # Show preview of next point
            if len(self.current_polygon) > 0:
                temp_frame = self.frame.copy()
                # Draw line from last point to current mouse position
                cv2.line(temp_frame, self.current_polygon[-1], (x, y), (0, 255, 255), 1)
                cv2.putText(temp_frame, f"Mouse: ({x}, {y})", (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.imshow("ROI Polygon Mapper", temp_frame)
    
    def draw_current_polygon(self):
        self.frame = self.original_frame.copy()
        
        # Draw all completed polygons
        for i, polygon in enumerate(self.polygons):
            pts = np.array(polygon, np.int32)
            cv2.fillPoly(self.frame, [pts], (0, 255, 0, 50))  # Semi-transparent fill
            cv2.polylines(self.frame, [pts], True, (0, 255, 0), 2)
            # Add polygon label
            if len(polygon) > 0:
                cv2.putText(self.frame, self.polygon_names[i], polygon[0],
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current polygon being created
        if len(self.current_polygon) > 0:
            # Draw points
            for point in self.current_polygon:
                cv2.circle(self.frame, point, 3, (0, 0, 255), -1)
            
            # Draw lines between points
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon, np.int32)
                cv2.polylines(self.frame, [pts], False, (0, 0, 255), 2)
        
        # Add instructions
        cv2.putText(self.frame, "Left click: Add point | Right click: Finish polygon", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.frame, "Press 's': Save | 'c': Clear current | 'r': Reset all | 'q': Quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.frame, f"Current polygon: {len(self.current_polygon)} points", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.frame, f"Completed polygons: {len(self.polygons)}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("ROI Polygon Mapper", self.frame)
    
    def finish_polygon(self):
        if len(self.current_polygon) >= 3:
            # Get polygon name from user
            polygon_name = input(f"\nEnter name for polygon {len(self.polygons) + 1} (or press Enter for default): ").strip()
            if not polygon_name:
                polygon_name = f"ROI_{len(self.polygons) + 1}"
            
            self.polygons.append(self.current_polygon.copy())
            self.polygon_names.append(polygon_name)
            print(f"Polygon '{polygon_name}' completed with {len(self.current_polygon)} points")
            print(f"Points: {self.current_polygon}")
            
            self.current_polygon = []
            self.draw_current_polygon()
    
    def clear_current(self):
        self.current_polygon = []
        self.draw_current_polygon()
        print("Current polygon cleared")
    
    def reset_all(self):
        self.current_polygon = []
        self.polygons = []
        self.polygon_names = []
        self.draw_current_polygon()
        print("All polygons reset")
    
    def save_polygons(self):
        if not self.polygons:
            print("No polygons to save!")
            return
        
        # Save original frame with polygons drawn
        final_frame = self.original_frame.copy()
        for i, polygon in enumerate(self.polygons):
            pts = np.array(polygon, np.int32)
            cv2.fillPoly(final_frame, [pts], (0, 255, 0, 80))
            cv2.polylines(final_frame, [pts], True, (0, 255, 0), 3)
            # Add polygon labels
            cv2.putText(final_frame, self.polygon_names[i], polygon[0],
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Save annotated image
        image_file = os.path.join(session_folder, "roi_regions_annotated.jpg")
        cv2.imwrite(image_file, final_frame)
        
        # Save original frame
        original_file = os.path.join(session_folder, "original_frame.jpg")
        cv2.imwrite(original_file, self.original_frame)
        
        # Save JSON data
        data = {
            "timestamp": timestamp,
            "video_file": video_path,
            "frame_size": {"width": self.original_frame.shape[1], "height": self.original_frame.shape[0]},
            "polygons": []
        }
        
        for i, polygon in enumerate(self.polygons):
            data["polygons"].append({
                "name": self.polygon_names[i],
                "points": polygon
            })
        
        json_file = os.path.join(session_folder, "roi_polygons.json")
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate and save Python code
        self.generate_code_file()
        
        print(f"\n" + "="*60)
        print(f"FILES SAVED TO: {session_folder}")
        print("="*60)
        print(f"✓ Original frame: original_frame.jpg")
        print(f"✓ Annotated image: roi_regions_annotated.jpg")
        print(f"✓ JSON data: roi_polygons.json")
        print(f"✓ Python code: roi_code_for_main.py")
        print(f"✓ Text summary: roi_summary.txt")
        print("="*60)
        
        for i, polygon in enumerate(self.polygons):
            print(f"  {self.polygon_names[i]}: {len(polygon)} points")
    
    def generate_code_file(self):
        # Generate Python code file
        code_file = os.path.join(session_folder, "roi_code_for_main.py")
        summary_file = os.path.join(session_folder, "roi_summary.txt")
        
        with open(code_file, 'w') as f:
            f.write("# ROI Polygon Code for YOLOv8 Traffic Detection\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Video: {video_path}\n")
            f.write(f"# Frame size: {self.original_frame.shape[1]}x{self.original_frame.shape[0]}\n\n")
            f.write("import cv2\nimport numpy as np\n\n")
            
            # Write polygon definitions
            for i, polygon in enumerate(self.polygons):
                f.write(f"# {self.polygon_names[i]} polygon points\n")
                f.write(f"{self.polygon_names[i].lower()}_points = {polygon}\n")
                f.write(f"{self.polygon_names[i].lower()}_polygon = np.array({self.polygon_names[i].lower()}_points, np.int32)\n\n")
            
            # Write helper function
            f.write("# Function to check if point is inside polygon\n")
            f.write("def is_inside_polygon(x, y, polygon):\n")
            f.write("    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0\n\n")
            
            # Write usage example
            f.write("# Usage in detection loop:\n")
            f.write("# Get center point of detected object\n")
            f.write("# center_x = int((x1 + x2) / 2)\n")
            f.write("# center_y = int((y1 + y2) / 2)\n\n")
            
            for i, polygon in enumerate(self.polygons):
                f.write(f"# Check if object is inside {self.polygon_names[i]}\n")
                f.write(f"# if is_inside_polygon(center_x, center_y, {self.polygon_names[i].lower()}_polygon):\n")
                f.write(f"#     print('Object detected in {self.polygon_names[i]}')\n\n")
        
        # Generate summary text file
        with open(summary_file, 'w') as f:
            f.write("ROI POLYGON MAPPING SUMMARY\n")
            f.write("="*40 + "\n\n")
            f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video File: {video_path}\n")
            f.write(f"Frame Size: {self.original_frame.shape[1]}x{self.original_frame.shape[0]}\n")
            f.write(f"Total Polygons: {len(self.polygons)}\n\n")
            
            for i, polygon in enumerate(self.polygons):
                f.write(f"POLYGON {i+1}: {self.polygon_names[i]}\n")
                f.write(f"Points ({len(polygon)}): {polygon}\n\n")
            
            f.write("INSTRUCTIONS:\n")
            f.write("-"*20 + "\n")
            f.write("1. Copy the polygon definitions from 'roi_code_for_main.py'\n")
            f.write("2. Replace the automatic ROI calculation in your main.py\n")
            f.write("3. Use the is_inside_polygon() function to check detections\n")
        
        # Also print to console
        print("\n" + "="*50)
        print("PYTHON CODE FOR YOUR MAIN SCRIPT:")
        print("="*50)
        
        for i, polygon in enumerate(self.polygons):
            print(f"\n# {self.polygon_names[i]} polygon points")
            print(f"{self.polygon_names[i].lower()}_points = {polygon}")
            print(f"{self.polygon_names[i].lower()}_polygon = np.array({self.polygon_names[i].lower()}_points, np.int32)")
        
        print(f"\n# Function to check if point is inside polygon")
        print(f"def is_inside_polygon(x, y, polygon):")
        print(f"    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0")
        
        print(f"\n# Usage in detection loop:")
        print(f"# center_x = int((x1 + x2) / 2)")
        print(f"# center_y = int((y1 + y2) / 2)")
        for i, polygon in enumerate(self.polygons):
            print(f"# if is_inside_polygon(center_x, center_y, {self.polygon_names[i].lower()}_polygon):")
            print(f"#     # Object is inside {self.polygon_names[i]}")
        print("="*50)

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found!")
    sys.exit(1)

# Open video and get first frame
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    cap.release()
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video loaded: {frame_width}x{frame_height}")

# Create ROI mapper
mapper = PolygonROIMapper(frame)

# Set up window and mouse callback
cv2.namedWindow("ROI Polygon Mapper", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ROI Polygon Mapper", mapper.mouse_callback)

# Initial display
mapper.draw_current_polygon()

print("\nROI Polygon Mapper Instructions:")
print("- Left click: Add point to current polygon")
print("- Right click: Finish current polygon (minimum 3 points)")
print("- Press 's': Save all polygons")
print("- Press 'c': Clear current polygon")
print("- Press 'r': Reset all polygons")
print("- Press 'q': Quit")

try:
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            # Auto-save before quitting if there are polygons
            if mapper.polygons:
                print("\nAuto-saving polygons before quit...")
                mapper.save_polygons()
            else:
                print("No polygons to save.")
            break
        elif key == ord('s'):
            mapper.save_polygons()
        elif key == ord('c'):
            mapper.clear_current()
        elif key == ord('r'):
            mapper.reset_all()

except KeyboardInterrupt:
    print("\nInterrupted by user")
    # Auto-save on Ctrl+C too
    if mapper.polygons:
        print("Auto-saving polygons...")
        mapper.save_polygons()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("ROI mapping completed!")