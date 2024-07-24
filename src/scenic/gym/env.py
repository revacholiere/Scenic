import gymnasium as gym
from scenic.simulators.carla.simulator import CarlaSimulator


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

        return obs, reward, done, info

    def reset(self):

        self.simulation = self.simulator.simulate(
            scene=self.scene, timestep=self.timestep
        )

        obs = self.simulation.getEgoImage()

        return obs

    def end_episode(self):
        self.simulation.destroy()

    def close(self):
        self.simulator.destroy()
