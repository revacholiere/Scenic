import gymnasium as gym
import random
import numpy as np
import scenic
import carla
import random


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
        
        
    
    

    def step(self, ctrl): # TODO: Add reward, observation, agent info
        info = {}
        done = False
        reward = 0

        
        
        self.simulation.setEgoControl(ctrl)
        self.simulation.run_one_step()

        obs = self.simulation.getEgoImage()
        
        
        return obs, reward, done, info

    def reset(self):
        self.simulation = self.simulator.simulate(scene = self.scene, timestep = self.timestep)
        obs = self.simulation.getEgoImage()
 
    
    

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
    

    env = CarlaEnv(scene = scene, carla_map = carla_map, map_path = map_path)
    
    obs = env.reset()
    obs.save_to_disk('images/%.6d.jpg' % obs.frame)
    print("reset the environment")
    for i in range(0):
        

        if i % 20 == 0:
            obs.save_to_disk('images/%.6d.jpg' % obs.frame)
        
        control = random_vehicle_control()
        obs = env.step(control)
    

    env.close()




