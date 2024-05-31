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
        self.simulation.step()

        
        
        

    def reset(self):
        self.simulation, _ = self.simulator.createSimulation(scene = self.scene, timestep = self.timestep, name="test")
        self.simulation.initialize_simulation()
        print("simulation created")
      
    
    

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
    map_path = scenic.syntax.veneer.localPath('~/Scenic/assets/maps/CARLA/Town01.xodr')
    carla_map = 'Town01'
    scenario = scenic.scenarioFromFile("test.scenic", mode2D = True)
    random.seed(0)
    scene, _ =  scenario.generate()
    print(scene)

    env = CarlaEnv(scene = scene, carla_map = carla_map, map_path = map_path)
    
    env.reset()
    print("reset the environment")
    for i in range(100):

        control = random_vehicle_control()
        env.step(control)

    env.close()
    

main()