import hashlib
import logging
import random
import pygame
import numpy as np
import torch
import carla
from config import cfg
from agents.tools.misc import get_speed
from object_info import ObjectInfo
logger = logging.getLogger(__name__)


def get_hash_seed(*args):
    # Convert all inputs to strings and concatenate them
    input_string = str(cfg.seed) + "".join(str(arg) for arg in args)

    # Hash the input string to get a fixed size output
    hash_object = hashlib.md5(input_string.encode())
    hash_hex = hash_object.hexdigest()

    # Convert the hexadecimal hash to an integer
    seed_int = int(hash_hex, 16)

    # To ensure the seed fits within 32-bit integer, take modulo 2**32
    final_seed = seed_int % (2**32)

    return final_seed


def set_seeds(seed=None):
    if seed is None:
        seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"SEED: {seed}")





def get_ground_truth_objects(vehicle, camera, distance_threshold=50):
    world = vehicle.get_world()
    logger.debug(f"vehicle_transform: {vehicle.get_transform()}")
    logger.debug(f"camera transform: {camera.get_transform()}")

    objects = []

    vehicles = world.get_actors().filter("*vehicle*")
    pedestrians = world.get_actors().filter("*walker.pedestrian*")
    npcs_and_categories = list(zip(vehicles, ["car"] * len(vehicles))) + list(
        zip(pedestrians, ["person"] * len(pedestrians))
    )

    for npc, category in npcs_and_categories:
        # Filter out the ego vehicle
        if npc.id == vehicle.id:
            logger.debug(f"Skipping ego vehicle: {npc}")
            continue

        # Filter for the vehicles within distance_threshold
        distance = npc.get_transform().location.distance(
            vehicle.get_transform().location
        )

        if distance < 1.0 or distance > distance_threshold:
            continue

        # Calculate the dot product between the forward vector
        # of the vehicle and the vector between the vehicle
        # and the other vehicle. We threshold this dot product
        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
        forward_vec = vehicle.get_transform().get_forward_vector()
        ray = npc.get_transform().location - vehicle.get_transform().location
        if forward_vec.dot(ray) < 0:
            continue

        npc_bbox_3d = npc.bounding_box

        world_verts = [
            [v.x, v.y, v.z] for v in npc_bbox_3d.get_world_vertices(npc.get_transform())
        ]

        # Project the 3D bounding box to the image
        image_points = project_points_world_to_image(world_verts, camera)
        x_min = min([p[0] for p in image_points])
        y_min = min([p[1] for p in image_points])
        x_max = max([p[0] for p in image_points])
        y_max = max([p[1] for p in image_points])

        speed = get_speed(npc)

        logger.debug(f"npc: {npc}")
        logger.debug(f"npc transform: {npc.get_transform()}")
        logger.debug(f"npc bounding box: {npc.bounding_box}")
        logger.debug(f"distance: {distance}")
        logger.debug(f"speed: {speed}")
        logger.debug(f"forward_vec: {forward_vec}")
        logger.debug(f"ray: {ray}")
        logger.debug(f"verts ({len(world_verts)}): {world_verts}")
        logger.debug(f"image_points ({len(image_points)}): {image_points}")
        logger.debug(f"bbox: {x_min}, {y_min}, {x_max}, {y_max}")

        obj = ObjectInfo(
            [x_min, y_min, x_max, y_max],
            category,
            1,
            npc.get_location(),
            distance,
            speed,
            npc_bbox_3d,
        )
        objects.append(obj)
    return objects


def build_projection_matrix_and_inverse(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    K_inv = np.linalg.inv(K)
    return K, K_inv


def clamp_point_to_image(point, image_width, image_height):
    x, y = point
    x = max(0, min(x, image_width))
    y = max(0, min(y, image_height))
    return x, y


def ue4_to_standard_coordinates(point):
    return np.array([point[1], -point[2], point[0]])


def standard_to_ue4_coordinates(point):
    return np.array([point[2], point[0], -point[1]])


def project_points_world_to_image(points, camera):
    image_width = int(camera.attributes["image_size_x"])
    image_height = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
    K, _ = build_projection_matrix_and_inverse(image_width, image_height, fov)
    w2c = np.array(camera.get_transform().get_inverse_matrix())

    # Add homogeneous coordinate
    points = [np.append(p, 1) for p in points]

    # Transform to camera coordinates
    points_camera = [np.dot(w2c, p) for p in points]

    # Transform to standard coordinates
    points_camera_standard = [ue4_to_standard_coordinates(p) for p in points_camera]
    points_image = [np.dot(K, p) for p in points_camera_standard]
    points_image_normalized = [p[0:2] / p[2] for p in points_image]
    return points_image_normalized


def reconstruct_points_image_depth_to_world(points, camera):
    image_width = int(camera.attributes["image_size_x"])
    image_height = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
    _, K_inv = build_projection_matrix_and_inverse(image_width, image_height, fov)
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


def compute_locations_and_bboxes_3d(depth_image_meters, bboxes_xyxy, camera):
    logger.debug(f"depth_image_meters.shape: {depth_image_meters.shape}")
    logger.debug(
        f"depth_image_meters (min, max): ({np.min(depth_image_meters)}, {np.max(depth_image_meters)})"
    )
    locations = []
    bboxes_3d = []
    for bbox in bboxes_xyxy:
        x1, y1, x2, y2 = bbox
        xc = int((x1 + x2) / 2)
        yc = int((y1 + y2) / 2)

        depth = depth_image_meters[yc, xc]
        logger.debug(f"depth: {depth}")

        points_2d = [
            (xc, yc, depth),
            (x1, y1, depth),
            (x1, y2, depth),
            (x2, y1, depth),
            (x2, y2, depth),
        ]
        logger.debug(f"points_2d: {points_2d}")
        points_3d = reconstruct_points_image_depth_to_world(points_2d, camera)
        logger.debug(f"points_3d: {points_3d}")
        bbox_3d_center = points_3d[0]
        bbox_3d_corners = points_3d[1:]

        # Use camera transform to get world coordinates
        # location_vec = camera.get_transform().transform(carla.Location(*bbox_3d_center))
        # logger.debug(f"location_vec: {location_vec}")
        # location = carla.Location(x=location_vec.x, y=location_vec.y, z=location_vec.z)

        #######################################################################################################
        # The bbox_3d_center already is in world coordinate system, we don't need to do transformation again  #
        # Above three lines code can be deleted                                                               #
        #######################################################################################################
        location = carla.Location(
            x=bbox_3d_center[0], y=bbox_3d_center[1], z=bbox_3d_center[2]
        )
        logger.debug(f"location: {location}")
        locations.append(location)

        # Compute 3D bounding box extents
        extents = np.max(bbox_3d_corners, axis=0) - np.min(bbox_3d_corners, axis=0)
        logger.debug(f"extents: {extents}")
        bbox_3d = carla.BoundingBox()
        bbox_3d.location = location
        bbox_3d.extent.x, bbox_3d.extent.y, bbox_3d.extent.z = (
            extents / 2
        )  # Assuming center-aligned bbox
        logger.debug(f"bbox_3d: {bbox_3d}")
        bboxes_3d.append(bbox_3d)

    return locations, bboxes_3d


def polar2xyz(points):
    """
    This method convert the radar point from polar coordinate to xyz coordinate
    :param points: points from the radar data with format([[vel, azimuth, altitude, depth],...[,,,]])
    :return: The xyz locations with carla.Location format
    """
    points_location = []
    az_cos = np.cos(points[:, 1])
    az_sin = np.sin(points[:, 1])
    al_cos = np.cos(points[:, 2])
    al_sin = np.sin(points[:, 2])
    points_x = np.multiply(np.multiply(points[:, 3], al_cos), az_cos)
    points_y = np.multiply(np.multiply(points[:, 3], al_cos), az_sin)
    points_z = np.multiply(points[:, 3], al_sin)
    for i in range(points_x.shape[0]):
        x = points_x[i]
        y = points_y[i]
        z = points_z[i]
        point_location = carla.Location(float(x), float(y), float(z))
        # point_location.x = x
        # point_location.y = y
        # point_location.z = z
        # point_location = carla.Location(x=0, y=0,z = 0)
        points_location.append(point_location)
    return points_location


def nearest_point(radar_point, object_location):
    """
    This method find the nearest point of each object_location
    :param radar_point: The point captured from radar(carla.Location)
    :param object_location: The object location in the simulated world(carla.Location)
    :return: The searched location index
    """
    indices = []
    for i in range(len(object_location)):
        min_dist = 10000
        index = 0
        for j in range(len(radar_point)):
            ol = object_location[i]
            rp = radar_point[j]
            dist = ol.distance(rp)
            if dist < min_dist:
                index = j
                min_dist = dist
        indices.append(index)
    return indices



def image_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(
        image.height, image.width, 4
    )
    array = array[:, :, :3]  # BGR
    array = array[:, :, ::-1]  # RGB
    return array


def image_to_grayscale_depth_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(
        image.height, image.width, 4
    )
    array = array.astype(np.float32)
    array = array[:, :, :3]  # BGR

    B, G, R = array[:, :, 0], array[:, :, 1], array[:, :, 2]

    # Apply the formula to normalize the depth values
    grayscale = ((R + G * 256 + B * 65536) / (16777215)) * 1000

    return np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)







def image_to_surface(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


def pil_to_surface(image):
    array = np.asarray(image)
    array = np.reshape(array, (image.height, image.width, 3))
    #array = array[:, :, :]
    #array = array[:, :, ::-1]
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


def ego_image_array(image):
        ego_image = image
        ego_image = np.reshape(
            np.copy(ego_image.raw_data), (ego_image.height, ego_image.width, 4)
        )
        ego_image = ego_image[:, :, 0:3]
        return ego_image