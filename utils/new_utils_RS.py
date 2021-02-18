'''
Date of modification    : 2021.01.22
Code Summary            : realsense camera python code ìµœì¢… ë²„ì „ 
Input                   option      0 : D435i (default)
	                                1 : L515
	                                2 : D445i
'''

import pyrealsense2 as rs
import numpy as np
import cv2

class Realsense:
    def __init__(self, opt):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.device_Option(opt)

        self.profile = self.pipeline.start(self.config)
        
        self.depth_scale = self.get_Depthscale()

        align_to = rs.stream.color
        self.align = rs.align(align_to)
    
    def device_Option(self, opt):  
        fps = 30
        if opt == 0:
            print('Realsense Device >> D435i')
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        elif opt == 1:
            print('Realsense Device >> L515')
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        else :
            print('Something wrong about Realsense Device !!')

    def get_Depthscale(self):
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        scaling = self.depth_sensor.get_depth_scale()
        return scaling

    def output_image(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
        self.depth_image = self.depth_image * self.depth_scale

        return color_image, self.depth_image

    def get_Intrinsics(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        self.rs_intrinsic = color_frame.profile.as_video_stream_profile().intrinsics

        return self.rs_intrinsic

    # input [x(width), y(height))]
    def get_Depth_Point(self, x_p, y_p):
        if x_p >= self.depth_image.shape[1]:
            print('ERROR : x_p(width) value ')
        if y_p >= self.depth_image.shape[0]:
            print('ERROR : y_p(height) value ')

        pixel = [x_p, y_p]
        depth_point = rs.rs2_deproject_pixel_to_point(self.rs_intrinsic, pixel, self.depth_image[y_p][x_p])

        return depth_point

