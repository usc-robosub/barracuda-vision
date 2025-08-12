#!/usr/bin/env python3

import numpy as np
import rospy

class CameraModel:
    """Camera model for 3D projection and depth calculations"""
    
    def __init__(self, horizontal_fov=1.089, width=1920, height=1080):
        """Initialize camera model with simulation parameters"""
        self.horizontal_fov = horizontal_fov  # radians
        self.width = width
        self.height = height
        
        # Calculate focal length from FOV
        self.fx = width / (2 * np.tan(horizontal_fov / 2))
        self.fy = self.fx  # Assume square pixels
        
        # Principal point at image center
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        # Create camera matrix
        self.camera_matrix = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1]
        ])
        
        rospy.loginfo(f"Camera model initialized: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
    
    def project_to_3d(self, pixel_x, pixel_y, depth):
        """Project 2D pixel coordinates to 3D camera coordinates"""
        # Convert pixel coordinates to normalized camera coordinates
        x_norm = (pixel_x - self.cx) / self.fx
        y_norm = (pixel_y - self.cy) / self.fy
        
        # Calculate the normalization factor for the depth distance
        # depth is the actual distance from camera, so we need to scale by the magnitude
        norm_factor = np.sqrt(1 + x_norm*x_norm + y_norm*y_norm)
        
        # Calculate 3D coordinates in camera frame
        x = depth / norm_factor  # Forward distance (along camera's Z-axis)
        y = -x_norm * depth / norm_factor  # Right direction (negative for correct orientation)
        z = -y_norm * depth / norm_factor  # Down direction (negative for correct orientation)
        
        return np.array([x, y, z])
