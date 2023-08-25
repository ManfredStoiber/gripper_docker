from cares_gripper.scripts.environments.Environment import Environment
import logging
import numpy as np

from pathlib import Path

file_path = Path(__file__).parent.resolve()

from cares_gripper.scripts.configurations import EnvironmentConfig, GripperConfig

from cares_lib.vision.Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.dynamixel.Servo import DynamixelServoError


##### Set goal functions

#####

# TODO turn the hard coded type ints into enums
class TranslationEnvironment(Environment):
    def __init__(self, env_config: EnvironmentConfig, gripper_config: GripperConfig):
        super().__init__(env_config, gripper_config)

    # overriding method
    def choose_goal(self):
        position = self.get_object_state()['position']
        position[0] = np.random.randint(-60, 45)
        position[1] = np.random.randint(35, 60)

        return position

    def reward_function(self, target_goal, goal_before, goal_after):
        if goal_before is None:
            logging.debug("Start Marker Pose is None")
            return 0, True

        if goal_after is None:
            logging.debug("Final Marker Pose is None")
            return 0, True
        done = False

        # update
        #############################################

        target_goal = target_goal[0:2]
        goal_after_array = goal_after["position"][0:2]
        goal_difference = np.linalg.norm(target_goal - goal_after_array)

        print(target_goal, goal_after_array, goal_difference)

        # logging.info(f"x after: {x_after}, y after: {y_after}")
        ############################################

        # The following step might improve the performance. But dont have to use for simple tasks like Translation.

        # goal_before_array = goal_before["position"][0:2]
        # delta_changes   = np.linalg.norm(target_goal - goal_before_array) - np.linalg.norm(target_goal - goal_after_array)
        # if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
        #     reward = -10
        # else:
        #     reward = -goal_difference
        #     #reward = delta_changes / (np.abs(yaw_before - target_goal))
        #     #reward = reward if reward > 0 else 0

        # For Translation. noise_tolerance is 15, it would affect the performance to some extent.
        if goal_difference <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            done = True
            reward = 500
        else:
            reward = -goal_difference

        return reward, done
