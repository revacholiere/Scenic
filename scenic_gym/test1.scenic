param map = localPath('../../Scenic/assets/maps/CARLA/Town01.xodr')
param carla_map = 'Town01'
param time_step = 1.0/10


model scenic.simulators.carla.model



ego = new Car

car2 = new Car visible from ego, with behavior AutopilotBehavior()
car3 = new Car visible from ego, with behavior AutopilotBehavior()
car4 = new Car visible, with behavior AutopilotBehavior()
car5 = new Car visible, with behavior AutopilotBehavior()
car6 = new Car with behavior AutopilotBehavior()
car7 = new Car with behavior AutopilotBehavior()
car8 = new Car with behavior AutopilotBehavior()
car9 = new Car with behavior AutopilotBehavior()
