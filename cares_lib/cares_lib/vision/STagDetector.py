from enum import Enum

import cv2
import math
import cares_lib.utils.utils as utils
import numpy as np
import pyStag as stag


class Dictionary(Enum):
    HD11 = 11
    HD13 = 13
    HD15 = 15
    HD17 = 17
    HD19 = 19
    HD21 = 21
    HD23 = 23


class STagDetector:
    def __init__(self, marker_size, dictionary=Dictionary.HD21):
        self.dictionary = dictionary
        self.error_correction = 7
        self.marker_size = marker_size
        self.t_vecs = []

    def calculate_target_position(self, goal_x_in_pixel, goal_y_in_pixel, camera_matrix):
        z  = self.t_vecs[0, 0, 2]
        
        cx = camera_matrix[0, 2]
        fx = camera_matrix[0, 0]
        cy = camera_matrix[1, 2]
        fy = camera_matrix[1, 1]

        px = (goal_x_in_pixel - cx) / fx
        py = (goal_y_in_pixel - cy) / fy

        px = px * z
        py = py * z

        target_point_wrt_camera = (px, py)
        return np.array(target_point_wrt_camera)


    def get_orientation(self, r_vec):
        r_matrix, _ = cv2.Rodrigues(r_vec)
        roll, pitch, yaw = utils.rotation_to_euler(r_matrix)

        def validate_angle(degrees):
            return degrees % 360

        roll  = validate_angle(math.degrees(roll))
        pitch = validate_angle(math.degrees(pitch))
        yaw   = validate_angle(math.degrees(yaw))

        return [roll, pitch, yaw]

    def get_pose(self, t_vec, r_vec):
        pose = {}
        pose["position"] = t_vec[0]
        pose["orientation"] = self.get_orientation(r_vec)
        return pose
        
    def get_marker_poses(self, image, camera_matrix, camera_distortion, display=True):
        self.detector = stag.Detector(self.dictionary.value, self.error_correction, False)
        marker_poses = {}

        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (corners, ids) = cv2.aruco.detectMarkers(grayImg)

        corners = [np.array(c) for c in corners]
        ids = np.array(ids).reshape(-1, 1)

        if len(corners) > 0:
            r_vecs, t_vecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, camera_matrix, camera_distortion)

            image_copy = image.copy()

            for bbx, id in zip(corners, ids):
                image_copy = cv2.line(image_copy, tuple(map(int, bbx[0])), tuple(map(int, bbx[1])), (255, 225, 0), 5)
                image_copy = cv2.line(image_copy, tuple(map(int, bbx[1])), tuple(map(int, bbx[2])), (255, 225, 0), 5)
                image_copy = cv2.line(image_copy, tuple(map(int, bbx[2])), tuple(map(int, bbx[3])), (255, 0, 225), 5)
                image_copy = cv2.line(image_copy, tuple(map(int, bbx[3])), tuple(map(int, bbx[0])), (255, 0, 225), 5)
                #image_copy = cv2.putText(image_copy, "{}".format(id), tuple(map(int, bbx[0])), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 225), 2)

            for i in range(0, len(r_vecs)):
                cv2.drawFrameAxes(image_copy, camera_matrix,
                               camera_distortion, r_vecs[i], t_vecs[i], self.marker_size/2.0, 3)

            self.t_vecs = t_vecs
            for i in range(0, len(r_vecs)):
                id = ids[i][0]
                r_vec = r_vecs[i]
                t_vec = t_vecs[i]
          	    #TODO: change this to output something less bulky than two arrays 
                marker_poses[id] = self.get_pose(t_vec, r_vec)

            if display:
                cv2.imshow("Frame", image_copy)
                cv2.waitKey(100)

        return marker_poses
