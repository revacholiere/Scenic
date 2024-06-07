import gymnasium as gym
import random
import numpy as np
import scenic
import carla
import random
from rulebook import RuleBook
from agents.navigation.behavior_agent import BehaviorAgent
from object_info import create_obj_list
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example values
])

def image_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(image.height, image.width, 4)
    array = array[:, :, :3] #BGR
    array = array[:, :, ::-1] #RGB
    return array





WIDTH = 1280
HEIGHT = 720


class CarlaEnv(gym.Env):
    def __init__(self, scene, carla_map,
        map_path,
        address="127.0.0.1",
        port=2000,
        timeout=10,
        render=True,
        record="",
        timestep=0.1,
        traffic_manager_port=None,
    ):
        
        #self.observation_space = gym.spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        #self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.timestep = timestep
        self.simulation = None
        self.scene = scene
        self.simulator = scenic.simulators.carla.simulator.CarlaSimulator(carla_map,
        map_path,
        address,
        port,
        timeout,
        render,
        record,
        timestep,
        traffic_manager_port)
        


        self.agent = None
        self.rulebook = None
        self.model = None
    
    

    def step(self, ctrl = None): # TODO: Add reward, observation, agent info
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
    
        # Get object detection
    
        pred = self.model(obs_array)
        pred_np = pred.cpu().numpy()
        
        # Update object list
        
        self.obj_list = create_obj_list(self.simulation, pred_np.boxes, depth_image) if len(pred[0].boxes) > 0 else []
        self.agent.update_object_information(self.obj_list)
        
        
        # Update rulebook
        
        self.rulebook.step()
        
        # Get ground truth bounding boxes
        
        
        
        
            
        

        
        return obs, reward, done, info

    def reset(self):
        self.simulation = self.simulator.simulate(scene = self.scene, timestep = self.timestep)


        obs = self.simulation.getEgoImage()
        #convert obs to numpy array


        
        self.agent = BehaviorAgent(self.simulation.ego, behavior='normal')
        self.rulebook = RuleBook(self.simulation.ego)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
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
        
        
        
def main(): # Test the environment
    
    
    
    map_path = scenic.syntax.veneer.localPath('~/Scenic/assets/maps/CARLA/Town01.xodr')
    carla_map = 'Town01'
    scenario = scenic.scenarioFromFile("test.scenic", mode2D = True)
    random.seed(2500)
    scene, _ =  scenario.generate()
    

    env = CarlaEnv(scene = scene, carla_map = carla_map, map_path = map_path, render = True)
    
    obs = env.reset()
    obs.save_to_disk('images/%.6d.jpg' % obs.frame)
    print("reset the environment")
    for i in range(100):
        

        obs.save_to_disk('images/%.6d.jpg' % obs.frame)
        
        obs = env.step()
    

    env.close()



main()
