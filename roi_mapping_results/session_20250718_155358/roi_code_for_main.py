# ROI Polygon Code for YOLOv8 Traffic Detection
# Generated on: 2025-07-18 15:54:59
# Video: intersectionRoad1.mp4
# Frame size: 640x360

import cv2
import numpy as np

# road1 polygon points
road1_points = [(286, 89), (469, 116), (500, 51), (339, 33), (285, 89)]
road1_polygon = np.array(road1_points, np.int32)

# road2 polygon points
road2_points = [(522, 158), (452, 308), (600, 336), (632, 186), (521, 158)]
road2_polygon = np.array(road2_points, np.int32)

# Function to check if point is inside polygon
def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

# Usage in detection loop:
# Get center point of detected object
# center_x = int((x1 + x2) / 2)
# center_y = int((y1 + y2) / 2)

# Check if object is inside road1
# if is_inside_polygon(center_x, center_y, road1_polygon):
#     print('Object detected in road1')

# Check if object is inside road2
# if is_inside_polygon(center_x, center_y, road2_polygon):
#     print('Object detected in road2')

