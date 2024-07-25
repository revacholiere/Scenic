param map = localPath('../../Scenic/assets/maps/CARLA/Town10HD.xodr')
param carla_map = 'Town10'
param time_step = 1.0/10


model scenic.simulators.carla.model



ego = new Car

for i in range(30):
    car = new Car with behavior AutopilotBehavior()
