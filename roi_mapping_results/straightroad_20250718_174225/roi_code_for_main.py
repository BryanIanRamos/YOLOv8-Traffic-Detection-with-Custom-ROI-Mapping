# ROI Polygon Code for YOLOv8 Traffic Detection
# Generated on: 2025-07-18 17:44:16
# Video: straightroad.mp4
# Frame size: 640x360

import cv2
import numpy as np

# right_road polygon points
right_road_points = [(383, 78), (276, 87), (412, 228), (638, 223), (628, 200), (383, 78)]
right_road_polygon = np.array(right_road_points, np.int32)

# left_road polygon points
left_road_points = [(209, 64), (94, 63), (7, 219), (292, 211), (211, 65)]
left_road_polygon = np.array(left_road_points, np.int32)

# Function to check if point is inside polygon
def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

# Usage in detection loop:
# Get center point of detected object
# center_x = int((x1 + x2) / 2)
# center_y = int((y1 + y2) / 2)

# Check if object is inside right_road
# if is_inside_polygon(center_x, center_y, right_road_polygon):
#     print('Object detected in right_road')

# Check if object is inside left_road
# if is_inside_polygon(center_x, center_y, left_road_polygon):
#     print('Object detected in left_road')

