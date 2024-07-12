import gymnasium as gym

# import random
# import numpy as np
# import scenic
from scenic.simulators.carla.simulator import CarlaSimulator

# import carla
# import random
# from rulebook import RuleBook
# import pygame

# import copy

# from object_info import create_obj_list

# from torchvision import transforms
# from ultralytics import YOLO

# from behavior_agent import BehaviorAgent
# from PIL import ImageDraw, Image

# from utils import image_to_array, image_to_grayscale_depth_array, draw_boxes, get_ground_truth_bboxes, image_to_surface, pil_to_surface
# from torchvision.utils import draw_bounding_boxes

# WIDTH = 1280
# HEIGHT = 720


class CarlaEnv(gym.Env):
    def __init__(
        self,
        carla_map,
        map_path,
        scene=None,
        address="127.0.0.1",
        port=2000,
        timeout=10,
        render=True,
        record="",
        timestep=0.1,
        traffic_manager_port=None,
    ):
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.timestep = timestep
        self.simulation = None
        self.scene = scene
        self.simulator = CarlaSimulator(
            carla_map,
            map_path,
            address,
            port,
            timeout,
            render,
            record,
            timestep,
            traffic_manager_port,
        )

    def set_scene(self, scene):
        self.scene = scene

    def getDepthImage(self):
        return self.simulation.getDepthImage()

    def getEgo(self):
        return self.simulation.ego

    def step(self, ctrl=None):  # TODO: Add reward, observation, agent info
        info = {}
        done = False
        reward = 0

        if ctrl is not None:
            self.simulation.setEgoControl(ctrl)

        self.simulation.run_one_step()

        obs = self.simulation.getEgoImage()
        # obs_array = image_to_array(obs)
        # depth_image = self.simulation.getDepthImage()
        # depth_array = image_to_grayscale_depth_array(depth_image)

        # Get object detection

        # pred = self.model(obs_array, verbose=False, classes=[1, 2, 3, 5, 7], conf=0.7)
        # pred_np = pred[0].cpu().numpy()

        # print(pred)

        # Update object list
        """
        self.obj_list = (
            create_obj_list(self.simulation, pred_np.boxes, depth_array)
            if len(pred[0].boxes) > 0
            else []
        )
        
        """

        # print(dir(pred))
        # self.agent.update_object_information(self.obj_list)

        # Update rulebook

        # self.rulebook.step()

        # Get ground truth bounding boxes

        return obs, reward, done, info

    def reset(self):

        self.simulation = self.simulator.simulate(
            scene=self.scene, timestep=self.timestep
        )

        obs = self.simulation.getEgoImage()

        # self.agent = BehaviorAgent(self.simulation.ego, behavior="normal")
        # self.rulebook = RuleBook(self.simulation.ego)
        # self.model = YOLO("./yolov5su.pt")
        # self.agent.update_object_information(self.obj_list)

        return obs

    def end_episode(self):
        self.simulation.destroy()
        self.simulator.client.reload_world()

    def close(self):
        self.simulator.destroy()
