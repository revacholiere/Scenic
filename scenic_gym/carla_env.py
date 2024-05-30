import gymnasium as gym
import random
from scenic.simulators import CarlaSimulator
from scenic.syntax.veneer import localPath
import scenic
import carla
import random

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

        
        self.scene = scene
        self.simulator = CarlaSimulator(carla_map,
        map_path,
        address,
        port,
        timeout,
        render,
        record,
        timestep,
        traffic_manager_port)
        
        
    
    

    def step(self, ctrl): # TODO: Add reward, observation, agent info
        

        
        
        self.simulation.setEgoControl(ctrl)
        self.simulation.step()

        
        
        

    def reset(self, seed=None):
        
        try:        
            self.simulation, _ = self.simulator.createSimulation(scene = self.scene, timestep = self.timestep)
            
        except Exception as e:
            print(e)
            #print("Retrying...")
            pass        
    
    
    

    def close(self):
        self.simulation.destroy()
        self.simulator.destroy()
        
        
        

def random_vehicle_control():
    control = carla.VehicleControl()
    control.throttle = random.uniform(0, 1)  # random throttle between 0 and 1
    control.steer = random.uniform(-1, 1)  # random steering angle between -1 and 1
    control.brake = random.uniform(0, 1)  # random brake between 0 and 1
    control.hand_brake = random.choice([False, True])  # random hand brake status
    control.reverse = random.choice([False, True])  # random reverse status
    return control
        
        
        
def main():
    map_path = localPath('~/Scenic/assets/maps/CARLA/Town01.xodr')
    carla_map = 'Town01'
    scenario = scenic.scenarioFromFile("test.scenic")
    
    env = CarlaEnv(scenario = scenario, carla_map = carla_map, map_path = map_path)
    
    env.reset()
    for i in range(100):

        control = random_vehicle_control()
        env.step(control)

    env.close()
    

    