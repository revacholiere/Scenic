#param map = localPath('~/Scenic/assets/maps/CARLA/Town01.xodr')
#param carla_map = 'Town01'
#param time_step = 1.0/10


model scenic.simulators.carla.model



ego = new Car
car = new Car in front of ego with behavior AutopilotBehavior()
