import gymnasium as gym
import random
import numpy as np
import scenic
from scenic.simulators.carla.simulator import CarlaSimulator
import carla
import random
from rulebook import RuleBook
import pygame

#import copy

from object_info import create_obj_list

from torchvision import transforms
from ultralytics import YOLO

from behavior_agent import BehaviorAgent
from PIL import ImageDraw, Image

from utils import image_to_array, image_to_grayscale_depth_array, draw_boxes, get_ground_truth_bboxes, image_to_surface, pil_to_surface
from torchvision.utils import draw_bounding_boxes

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


def random_vehicle_control():
    control = carla.VehicleControl()
    control.throttle = 1  # random throttle between 0 and 1
    control.steer = 0  # random steering angle between -1 and 1
    control.brake = 0  # random brake between 0 and 1
    control.hand_brake = False  # random hand brake status
    control.reverse = 0  # random reverse status
    return control


def main(seed, num_episodes, render = False):  # Test the environment
    map_path = scenic.syntax.veneer.localPath("~/Scenic/assets/maps/CARLA/Town01.xodr")
    carla_map = "Town01"
    env = CarlaEnv(carla_map=carla_map, map_path=map_path, render=False)
    model = YOLO("./yolov5su.pt")

    # scenario = scenic.scenarioFromFile("test.scenic", mode2D=True)
    if render:
            pygame.init()
            display = pygame.display.set_mode(
                (1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF
            )


    for j in range(num_episodes):

        scenario = scenic.scenarioFromFile(f"test1.scenic", mode2D=True)
        random.seed(seed + j)
        scene, _ = scenario.generate()
        env.set_scene(scene)
        obs = env.reset()

        agent = BehaviorAgent(env.getEgo(), behavior="normal")
        rulebook = RuleBook(env.getEgo())
        obj_list = []
        agent.update_object_information(obj_list)
        

        

        for i in range(100):  # number of timesteps

            obs_array = image_to_array(obs)
            depth_image = env.getDepthImage()
            depth_array = image_to_grayscale_depth_array(depth_image)

            pred = model(obs_array, verbose=False, classes=[1, 2, 3, 5, 7], conf=0.7)
            pred_np = pred[0].cpu().numpy()
            img = Image.fromarray(pred[0].plot())
            bboxes = get_ground_truth_bboxes(env.simulation.world.get_actors().filter('*vehicle*'), env.getEgo(), env.simulation.ego_camera.sensor)
            if len(bboxes) > 0 : draw_boxes(img, bboxes[:,1:][:,:4], (255, 0, 0, 100))

            obj_list = (
                create_obj_list(env.simulation, pred_np.boxes, depth_array)
                if len(pred[0].boxes) > 0
                else []
            )

            # print(pred)
            agent.update_object_information(obj_list)

            rulebook.step()

            ctrl = agent.run_step()

            obs, _, __, ___ = env.step(ctrl)

            if i % 10 == 0:
                #img.save_to_disk("seed%d_video%d/%.6d.jpg" % (seed, j, obs.frame))
                #img.save("images/seed%d_video%d%.6d.jpg" % (seed, j, obs.frame))
                #print(type(env.simulation.ego_camera._surface))
                #print(type(obs), type(img))
                surf_raw = image_to_surface(obs)
                surf_boxes = pil_to_surface(img)
                
                
                
                
                #pygame.image.save(surf_raw, "images/surf_raw%d.jpg" % (obs.frame))
                #pygame.image.save(surf_boxes, "images/surf_boxes%d.jpg" % (obs.frame))
                if render:
                    display.blit(surf_boxes, (0, 0))
                    pygame.display.flip()
                    

        print(f"end episode {j}")
        env.end_episode()

    env.close()


main(0, 1)
