param map = localPath('../../Scenic/assets/maps/CARLA/Town01.xodr')
param carla_map = 'Town01'
param time_step = 1.0/10


model scenic.simulators.carla.model



ego = new Car
car = new Car visible from ego, with behavior FollowLaneBehavior()
car2 = new Car visible from ego, with behavior AutoPilotBehavior()
car3 = new Car visible from ego, with behavior AutoPilotBehavior()
