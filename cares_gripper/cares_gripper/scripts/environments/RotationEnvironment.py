from cares_gripper.scripts.environments.Environment import Environment

import logging
import numpy as np

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from cares_gripper.scripts.configurations import EnvironmentConfig, GripperConfig

# from cares_lib.vision.Camera import Camera
# from cares_lib.vision.ArucoDetector import ArucoDetector
# from cares_lib.dynamixel.Servo import DynamixelServoError


def fixed_goal():
    target_angle = np.random.randint(1, 5)
    if target_angle == 1:
        return 90
    elif target_angle == 2:
        return 180
    elif target_angle == 3:
        return 270
    elif target_angle == 4:
        return 0
    raise ValueError(f"Target angle unknown: {target_angle}")
def fixed_goals(object_current_pose, noise_tolerance):
    current_yaw = object_current_pose['orientation'][2]# Yaw
    target_angle = fixed_goal()
    while abs(current_yaw - target_angle) < noise_tolerance:
        target_angle = fixed_goal()
    return target_angle

def relative_goal(current_target):
    return current_target + 90 #TODO redo this
#####

# TODO turn the hard coded type ints into enums
class RotationEnvironment(Environment):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig):
        super().__init__(env_config, gripper_config)

    # overriding method
    def choose_goal(self):
        if self.goal_selection_method == 0:# TODO Turn into enum
            return fixed_goals(self.get_object_state(), self.noise_tolerance)
        elif self.goal_selection_method == 1:
            return relative_goal(self.get_object_state())
        
        raise ValueError(f"Goal selection method unknown: {self.goal_selection_method}")

    def min_difference(self, a, b):
        return min(abs(a - b), (360 + min(a, b) - max(a, b)))

    # overriding method 
    def reward_function(self, target_goal, goal_before, goal_after):
        if goal_before is None: 
            logging.debug("Start Marker Pose is None")
            return 0, True

        if goal_after is None:
            logging.debug("Final Marker Pose is None")
            return 0, True
        
        done = False
        yaw_before = goal_before["orientation"][2]
        yaw_after  = goal_after["orientation"][2]

        goal_difference = self.min_difference(target_goal, yaw_after)
        delta_changes   = self.min_difference(target_goal, yaw_before) - self.min_difference(target_goal, yaw_after)

        #logging.info(f"Yaw = {yaw_after}")

        if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
            reward = -10
        else:
            reward = delta_changes / self.min_difference(target_goal, yaw_before)
            #reward = reward if reward > 0 else -10

        precision_range = 10
        if goal_difference <= precision_range:
            logging.info("----------Reached the Goal!----------")
            bonus = 100
            reward = reward + bonus
            done = True

        return reward, done
