import numpy as np
from commons import build_projection_matrix_and_inverse
from carla import Location, BoundingBox

def compute_locations_and_bboxes_3d(depth_array, bboxes_xyxy, camera):

        locations = []
        bboxes_3d = []
        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = bbox
            xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)

            depth = depth_array[yc, xc]

            points_2d = [
                (xc, yc, depth),
                (x1, y1, depth),
                (x1, y2, depth),
                (x2, y1, depth),
                (x2, y2, depth),
            ]
            
            points_3d = reconstruct_points_image_depth_to_world(points_2d, camera)
            bbox_3d_center = points_3d[0]
            bbox_3d_corners = points_3d[1:]

            location = Location(
                x=bbox_3d_center[0], y=bbox_3d_center[1], z=bbox_3d_center[2]
            )
            locations.append(location)

            # Compute 3D bounding box extents
            extents = np.max(bbox_3d_corners, axis=0) - np.min(bbox_3d_corners, axis=0)
            bbox_3d = BoundingBox()
            bbox_3d.location = location
            bbox_3d.extent.x, bbox_3d.extent.y, bbox_3d.extent.z = (
                extents / 2
            )  # Assuming center-aligned bbox

            bboxes_3d.append(bbox_3d)

        return locations, bboxes_3d

def reconstruct_points_image_depth_to_world(points, camera):
    image_width = int(camera.attributes["image_size_x"])
    image_height = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
    K_inv = build_projection_matrix_and_inverse(image_width, image_height, fov, True)
    c2w = np.array(camera.get_transform().get_matrix())

    points_world = []
    for x, y, depth in points:
        # Unproject to camera space
        point_camera_homogeneous = np.dot(K_inv, np.array([x, y, 1]))
        point_camera = point_camera_homogeneous[:3] * depth  # Scale by depth

        point_camera_ue4 = standard_to_ue4_coordinates(point_camera)
        point_world_homogeneous = np.dot(c2w, np.append(point_camera_ue4, 1))
        point_world = point_world_homogeneous[:3] / point_world_homogeneous[3]
        points_world.append(point_world)

    return points_world

def standard_to_ue4_coordinates(point):
    return np.array([point[2], point[0], -point[1]])

def create_obj_list(sim_manager, result_boxes, depth_array):

    bboxes_xyxy, classes = result_boxes.xyxy.tolist(), result_boxes.cls.tolist()
    depth_camera = sim_manager.depth_camera.sensor
    locations, bboxes_3d = compute_locations_and_bboxes_3d(depth_array, bboxes_xyxy, depth_camera)
    distances = [sim_manager.ego.get_location().distance(l) for l in locations]
    speeds = [0 if category==0 else 10 for category in classes]
    obj_list = [
        ObjectInfo(
            bbox, int(cls), location, distance, bbox_3d, speed
        )
    for bbox, cls , location, distance, bbox_3d, speed in zip(
        bboxes_xyxy,
        classes,
        locations,
        distances, 
        bboxes_3d, 
        speeds,
        )
    ]
    
    return obj_list
    
class ObjectInfo:
    # def __init__(self, bbox, category, confidence, location, distance, speed, bbox_3d):
    def __init__(self, bbox, category, location, distance, bbox_3d, speed):
        self._bbox = bbox
        self._category = category
        # self._confidence = confidence

        # estimated world location of the obj
        self._location = location
        self._distance = distance
        self._speed = speed
        self._bbox3D = bbox_3d
    
    @property
    def bbox_xyxy(self):
        return self._bbox

    @bbox_xyxy.setter
    def bbox_xyxy(self, bbox):
        self._bbox = bbox

    @bbox_xyxy.deleter
    def bbox_xyxy(self):
        del self._bbox

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, category):
        self._category = category

    @category.deleter
    def category(self):
        del self._category

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        self._location = location

    @location.deleter
    def location(self):
        del self._location

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance

    @distance.deleter
    def distance(self):
        del self._distance

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = speed

    @speed.deleter
    def speed(self):
        del self._speed

    @property
    def bbox_3d(self):
        return self._bbox3D

    @bbox_3d.setter
    def bbox_3d(self, bbox3D):
        self._bbox3D = bbox3D

    @bbox_3d.deleter
    def bbox_3d(self):
        del self._bbox3D

    # @property
    # def confidence(self):
    #     return self._confidence

    # @confidence.setter
    # def confidence(self, confidence):
    #     self._confidence = confidence

    # @confidence.deleter
    # def confidence(self):
    #     del self._confidence
