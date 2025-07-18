# ROI Polygon Code for YOLOv8 Traffic Detection
# Generated on: 2025-07-18 17:21:20
# Video: straightroad.mp4
# Frame size: 640x360

import cv2
import numpy as np

# Road1 polygon points
road1_points = [(340, 107), (308, 118), (413, 225), (497, 225), (355, 104), (340, 107)]
road1_polygon = np.array(road1_points, np.int32)

# road2 polygon points
road2_points = [(379, 101), (355, 105), (498, 227), (584, 226), (595, 224), (535, 172), (442, 130), (387, 98), (379, 102)]
road2_polygon = np.array(road2_points, np.int32)

# road3 polygon points
road3_points = [(422, 99), (394, 101), (568, 227), (636, 222), (631, 202), (633, 188), (547, 146), (438, 94), (422, 100)]
road3_polygon = np.array(road3_points, np.int32)

# road4 polygon points
road4_points = [(171, 69), (200, 225), (291, 214), (206, 59), (170, 68)]
road4_polygon = np.array(road4_points, np.int32)

# road5 polygon points
road5_points = [(129, 71), (172, 69), (200, 226), (105, 228), (128, 72)]
road5_polygon = np.array(road5_points, np.int32)

# road6 polygon points
road6_points = [(104, 228), (5, 222), (93, 58), (130, 71), (104, 231)]
road6_polygon = np.array(road6_points, np.int32)

# Function to check if point is inside polygon
def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

# Usage in detection loop:
# Get center point of detected object
# center_x = int((x1 + x2) / 2)
# center_y = int((y1 + y2) / 2)

# Check if object is inside Road1
# if is_inside_polygon(center_x, center_y, road1_polygon):
#     print('Object detected in Road1')

# Check if object is inside road2
# if is_inside_polygon(center_x, center_y, road2_polygon):
#     print('Object detected in road2')

# Check if object is inside road3
# if is_inside_polygon(center_x, center_y, road3_polygon):
#     print('Object detected in road3')

# Check if object is inside road4
# if is_inside_polygon(center_x, center_y, road4_polygon):
#     print('Object detected in road4')

# Check if object is inside road5
# if is_inside_polygon(center_x, center_y, road5_polygon):
#     print('Object detected in road5')

# Check if object is inside road6
# if is_inside_polygon(center_x, center_y, road6_polygon):
#     print('Object detected in road6')

