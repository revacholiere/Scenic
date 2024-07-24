import numpy as np
from PIL import ImageDraw, Image
import pygame



def build_projection_matrix_and_inverse(w, h, fov, inv=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    if inv: 
        return np.linalg.inv(K)
    else:
        return K

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]



def get_ground_truth_bboxes(actor_list, vehicle, camera):
    result= []
    
    image_w = int(camera.attributes["image_size_x"])
    image_h = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
        
    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix_and_inverse(image_w, image_h, fov, False)
    # Get the camera matrix 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    image_area = image_w * image_h
    
    for npc in actor_list:

        # Filter out the ego vehicle
        if npc.id == vehicle.id:
            continue

        bb = npc.bounding_box
        dist = npc.get_transform().location.distance(vehicle.get_transform().location)
        
        # Filter for the vehicles within 50m
        if dist < 30:
        # Calculate the dot product between the forward vector
        # of the vehicle and the vector between the vehicle
        # and the other vehicle. We threshold this dot product
        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = vehicle.get_transform().get_forward_vector()
            ray = npc.get_transform().location - vehicle.get_transform().location

            if forward_vec.dot(ray) > 1:
                p1 = get_image_point(bb.location, K, world_2_camera)
                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                #verts = np.array([[v.x, v.y, v.z, 1] for v in verts])
                #pts = get_image_point(verts, K, world_2_camera)
                
                x_max = -10000
                x_min = 10000
                y_max = -10000
                y_min = 10000

                for vert in verts:
                    p = get_image_point(vert, K, world_2_camera)
                    # Find the rightmost vertex
                    if p[0] > x_max:
                        x_max = p[0]
                    # Find the leftmost vertex
                    if p[0] < x_min:
                        x_min = p[0]
                    # Find the highest vertex
                    if p[1] > y_max:
                        y_max = p[1]
                    # Find the lowest  vertex
                    if p[1] < y_min:
                        y_min = p[1]
                
                
                
                #x_min, y_min, x_max, y_max = np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1])
                #x_min, y_min = np.clip(x_min, 0, image_w), np.clip(y_min, 0, image_h)
                #x_max, y_max = np.clip(x_max, 0, image_w), np.clip(y_max, 0, image_h)

                bbox_area = (x_max - x_min) * (y_max - y_min)
                
                if bbox_area <= 0.6 * image_area:
                    result.append([npc.id, x_min, y_min, x_max, y_max, 1, 0])

    return np.array(result)








def draw_boxes(img, boxes, outline=(0, 0, 255, 50)):

    img = ImageDraw.Draw(img, 'RGBA')

    for b in boxes:
        img.rectangle(list(b), fill=(0, 0, 255, 20), outline=outline, width=5)




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