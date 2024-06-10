import gymnasium as gym
import random
import numpy as np
import scenic
import carla
import random
from rulebook import RuleBook

from object_info import create_obj_list

from torchvision import transforms
from ultralytics import YOLO

from agent.navigation.behavior_agent import BehaviorAgent

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Example values
    ]
)


def image_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(
        image.height, image.width, 4
    )
    array = array[:, :, :3]  # BGR
    array = array[:, :, ::-1]  # RGB
    return array


def image_to_grayscale_depth_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(
        image.height, image.width, 4
    )
    array = array.astype(np.float32)
    array = array[:, :, :3]  # BGR

    B, G, R = array[:, :, 0], array[:, :, 1], array[:, :, 2]

    # Apply the formula to normalize the depth values
    grayscale = ((R + G * 256 + B * 65536) / (16777215)) * 1000

    return np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)


WIDTH = 1280
HEIGHT = 720


class CarlaEnv(gym.Env):
    def __init__(
        self,
        scene,
        carla_map,
        map_path,
        address="127.0.0.1",
        port=2000,
        timeout=10,
        render=True,
        record="",
        timestep=0.1,
        traffic_manager_port=None,
    ):
        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.timestep = timestep
        self.simulation = None
        self.scene = scene
        self.simulator = scenic.simulators.carla.simulator.CarlaSimulator(
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

        self.agent = None
        self.rulebook = None
        self.model = None
        self.obj_list = []

    def step(self, ctrl=None):  # TODO: Add reward, observation, agent info
        info = {}
        done = False
        reward = 0

        if ctrl is not None:
            self.simulation.setEgoControl(ctrl)
        else:
            self.simulation.setEgoControl(self.agent.run_step())

        self.simulation.run_one_step()

        obs = self.simulation.getEgoImage()
        obs_array = image_to_array(obs)
        depth_image = self.simulation.getDepthImage()
        depth_array = image_to_grayscale_depth_array(depth_image)

        # Get object detection

        pred = self.model(obs_array, verbose=False, classes=[1, 2, 3, 5, 7], conf=0.7)
        pred_np = pred[0].cpu().numpy()

        # print(pred)

        # Update object list

        self.obj_list = (
            create_obj_list(self.simulation, pred_np.boxes, depth_array)
            if len(pred[0].boxes) > 0
            else []
        )
        # print(dir(pred))
        self.agent.update_object_information(self.obj_list)

        # Update rulebook

        self.rulebook.step()

        # Get ground truth bounding boxes

        return obs, reward, done, info

    def reset(self):
        self.simulation = self.simulator.simulate(
            scene=self.scene, timestep=self.timestep
        )

        obs = self.simulation.getEgoImage()
        # convert obs to numpy array

        self.agent = BehaviorAgent(self.simulation.ego, behavior="normal")
        self.rulebook = RuleBook(self.simulation.ego)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model = YOLO("./yolov5s.pt")
        self.agent.update_object_information(self.obj_list)

        return obs

    def close(self):
        self.simulation.destroy()
        self.simulator.destroy()


def random_vehicle_control():
    control = carla.VehicleControl()
    control.throttle = 1  # random throttle between 0 and 1
    control.steer = 0  # random steering angle between -1 and 1
    control.brake = 0  # random brake between 0 and 1
    control.hand_brake = False  # random hand brake status
    control.reverse = 0  # random reverse status
    return control


def main(seed):  # Test the environment
    map_path = scenic.syntax.veneer.localPath("~/Scenic/assets/maps/CARLA/Town01.xodr")
    carla_map = "Town01"
    scenario = scenic.scenarioFromFile("test.scenic", mode2D=True)
    random.seed(seed)
    scene, _ = scenario.generate()

    env = CarlaEnv(scene=scene, carla_map=carla_map, map_path=map_path, render=True)

    obs = env.reset()

    print("reset the environment")
    for i in range(600):
        # print(i)
        obs, _, __, ___ = env.step()
        obs.save_to_disk("autopilotvideo%d/%d_%.6d.jpg" % (seed, seed, obs.frame))

    env.close()


main(3)
