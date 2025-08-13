#!/usr/bin/env python3

import numpy as np

class GeometryUtils:
    """Utility functions for 3D geometry calculations"""
    
    @staticmethod
    def bbox_center(x, y, width, height):
        """Calculate the center point of a bounding box"""
        center_x = x + width / 2
        center_y = y + height / 2
        return center_x, center_y
    
    @staticmethod
    def calculate_depth_at_point(depth_image, x, y, window_size=5):
        """Calculate median depth at a given point with error handling"""
        if depth_image is None:
            return None
        
        h, w = depth_image.shape
        x_int, y_int = int(x), int(y)
        
        # Check bounds
        if x_int < 0 or x_int >= w or y_int < 0 or y_int >= h:
            return None
        
        # Define sampling window
        half_window = window_size // 2
        x_min = max(0, x_int - half_window)
        x_max = min(w, x_int + half_window + 1)
        y_min = max(0, y_int - half_window)
        y_max = min(h, y_int + half_window + 1)
        
        # Extract depth values in window
        depth_window = depth_image[y_min:y_max, x_min:x_max]
        valid_depths = depth_window[depth_window > 0]  # Remove invalid (zero) depths
        
        if len(valid_depths) == 0:
            return None
        
        return float(np.median(valid_depths))
    
    @staticmethod
    def find_closest_point_in_bbox(depth_image, xmin, ymin, xmax, ymax):
        """Find the closest point (minimum depth) within a bounding box"""
        if depth_image is None:
            return None
        
        h, w = depth_image.shape
        
        # Convert to integers and ensure bounds
        xmin_int = max(0, int(xmin))
        ymin_int = max(0, int(ymin))
        xmax_int = min(w, int(xmax))
        ymax_int = min(h, int(ymax))
        
        # Check if bounding box is valid
        if xmin_int >= xmax_int or ymin_int >= ymax_int:
            return None
        
        # Extract depth values in bounding box
        bbox_depth = depth_image[ymin_int:ymax_int, xmin_int:xmax_int]
        
        # Flatten the array and remove invalid (zero) depths
        flat_depths = bbox_depth.flatten()
        valid_depths = flat_depths[flat_depths > 0]
        
        if len(valid_depths) == 0:
            return None
        
        # Find minimum depth
        min_depth = float(np.min(valid_depths))
        
        return min_depth
    
    @staticmethod
    def distance_3d(p1, p2):
        """Calculate Euclidean distance between two 3D points"""
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    @staticmethod
    def pose_to_dict(position, orientation=None):
        """Convert pose to dictionary format"""
        pose_dict = {
            'position': {
                'x': float(position[0]),
                'y': float(position[1]),
                'z': float(position[2])
            }
        }
        
        if orientation is not None:
            pose_dict['orientation'] = {
                'x': float(orientation[0]),
                'y': float(orientation[1]),
                'z': float(orientation[2])
            }
        
        return pose_dict
