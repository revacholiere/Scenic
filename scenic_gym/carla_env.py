import gymnasium as gym
import random

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
        

        
        
        self.simulation.setEgoControl(ctrl)
        self.simulation.run_one_step()

        
        
        

    def reset(self):
        self.simulation = self.simulator.simulate(scene = self.scene, timestep = self.timestep)
 
    
    

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
        
        
        
def main():
    map_path = scenic.syntax.veneer.localPath('~/Scenic/assets/maps/CARLA/Town01.xodr')
    carla_map = 'Town01'
    scenario = scenic.scenarioFromFile("test.scenic", mode2D = True)
    random.seed(2)
    scene, _ =  scenario.generate()
    

    env = CarlaEnv(scene = scene, carla_map = carla_map, map_path = map_path)
    
    env.reset()
    print("reset the environment")
    for i in range(1000):
        
        print(i)
        control = random_vehicle_control()
        env.step(control)
        print('after')

    env.close()
    

main()