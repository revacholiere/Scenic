import logging
import os
from string import Template

import carla
import numpy as np
import torch
from carla import ColorConverter as cc
from PIL import Image
from PIL import ImageFile

from object_info import ObjectInfo


ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import cfg
from logger_config import SUPPRESS


logger = logging.getLogger(__name__)
pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(SUPPRESS)

path_templates = {
    "vehicle_state": Template(
        "/result/${experiment_name}/vehicle/state/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "vehicle_control": Template(
        "/result/${experiment_name}/vehicle/control/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "score": Template(
        "/result/${experiment_name}/rewards/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "ego_image": Template(
        "/result/${experiment_name}/images/ego/${sample_or_baseline}_${epoch}_${trajectory}_${step}.png"
    ),
    "birdseye_image": Template(
        "/result/${experiment_name}/images/birdseye/${sample_or_baseline}_${epoch}_${trajectory}_${step}.png"
    ),
    "depth_image_raw": Template(
        "/result/${experiment_name}/images/depth_raw/${sample_or_baseline}_${epoch}_${trajectory}_${step}.png"
    ),
    "depth_image_log": Template(
        "/result/${experiment_name}/images/depth_log/${sample_or_baseline}_${epoch}_${trajectory}_${step}.png"
    ),
    "predicted_objects": Template(
        "/result/${experiment_name}/predictions/objects/${sample_or_baseline}_${epoch}_${trajectory}_${step}.txt"
    ),
    "predicted_captions": Template(
        "/result/${experiment_name}/predictions/captions/${sample_or_baseline}_${epoch}_${trajectory}_${step}.txt"
    ),
    "ground_truth_objects": Template(
        "/result/${experiment_name}/ground_truth/objects/${sample_or_baseline}_${epoch}_${trajectory}_${step}.txt"
    ),
    "ego_video": Template(
        "/result/${experiment_name}/videos/ego/${sample_or_baseline}_${epoch}_${trajectory}.mp4"
    ),
    "birdseye_video": Template(
        "/result/${experiment_name}/videos/birdseye/${sample_or_baseline}_${epoch}_${trajectory}.mp4"
    ),
    "percep_reward": Template(
        "/result/${experiment_name}/percep_reward/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "percep_recall": Template(
        "/result/${experiment_name}/percep_recall/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "rb_reward": Template(
        "/result/${experiment_name}/rewards/${sample_or_baseline}_${epoch}_${trajectory}.txt"
    ),
    "metrics": Template("/result/${experiment_name}/metrics/${epoch}.json"),
}


def get_indices_from_filename(filename):
    filename = filename.split("/")[-1]
    filename = filename.split(".")[0]
    filename = filename.split("_")
    epoch = int(filename[1])
    trajectory = int(filename[2])
    if len(filename) > 3:
        step = int(filename[3])
        return epoch, trajectory, step
    else:
        return (
            epoch,
            trajectory,
        )


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def make_permissive_dir(dir):
    if not os.path.exists(dir):
        old_umask = os.umask(0)
        os.makedirs(dir, mode=0o777, exist_ok=True)
        os.umask(old_umask)


def fill_template(template, **kwargs):
    return template.substitute(**kwargs)


def sample_or_baseline(is_baseline):
    if is_baseline:
        return "baseline"
    else:
        return "sample"


def save_objects(objects, path):
    dir = os.path.dirname(path)
    make_dir(dir)

    with open(path, "w") as file:
        for object in objects:
            bbox_xyxy = object.bbox_xyxy
            category = object.category
            confidence = object.confidence
            location = object.location
            distance = object.distance
            speed = object.speed
            bbox_3d = " ".join(
                map(
                    str,
                    [
                        object.bbox_3d.location.x,
                        object.bbox_3d.location.y,
                        object.bbox_3d.location.z,
                        object.bbox_3d.extent.x,
                        object.bbox_3d.extent.y,
                        object.bbox_3d.extent.z,
                        object.bbox_3d.rotation.pitch,
                        object.bbox_3d.rotation.yaw,
                        object.bbox_3d.rotation.roll,
                    ],
                )
            )
            file.write(
                f"{' '.join(map(str, [*bbox_xyxy, category, confidence, location.x, location.y, location.z, distance, speed, bbox_3d]))}\n"
            )
    return path


def load_objects(path):
    objects = []
    with open(path, "r") as file:
        for line in file.readlines():
            bbox_xyxy = [
                int(round(float(item))) for item in line.strip().split(" ")[:4]
            ]
            category = line.strip().split(" ")[4]
            score = float(line.strip().split(" ")[5])
            location = (
                float(line.strip().split(" ")[6]),
                float(line.strip().split(" ")[7]),
                float(line.strip().split(" ")[8]),
            )
            distance = float(line.strip().split(" ")[9])
            speed = float(line.strip().split(" ")[10])
            bbox_3d_list = line.strip().split(" ")[11:]
            bbox_3d_list = list(map(float, bbox_3d_list))
            if len(bbox_3d_list) == 9:
                bbox_3d = carla.BoundingBox(
                    carla.Location(*bbox_3d_list[:3]),
                    carla.Vector3D(*bbox_3d_list[3:6]),
                )
                bbox_3d.rotation = carla.Rotation(*bbox_3d_list[6:])
            else:
                bbox_3d = None
            object = ObjectInfo(
                bbox_xyxy, category, score, location, distance, speed, bbox_3d
            )
            objects.append(object)
    return objects


def save_image(image, path):
    dir = os.path.dirname(path)
    make_dir(dir)
    # Reorder channels from BGRA to RGBA
    rgba_image = image[..., [2, 1, 0]]
    img_save = Image.fromarray(rgba_image.astype("uint8"))
    img_save.save(path)
    return path


def load_image(path):
    image = Image.open(path)
    # Convert image to numpy array
    image_np = np.array(image)
    bgr_image = image_np[..., [2, 1, 0]]
    return bgr_image


def save_predicted_objects(objects, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["predicted_objects"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return save_objects(objects, path)


def load_predicted_objects(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["predicted_objects"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_objects(path)


def save_predicted_captions(captions, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["predicted_captions"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        for caption in captions:
            file.write(caption + "\n")
    return path


def load_predicted_captions(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["predicted_captions"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    captions = []
    with open(path, "r") as file:
        for line in file.readlines():
            caption = line.strip()
            captions.append(caption)
    return captions


def save_ground_truth_objects(objects, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["ground_truth_objects"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return save_objects(objects, path)


def load_ground_truth_objects(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["ground_truth_objects"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_objects(path)

def save_scores_boxwise(scores, epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["score"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        for name, sublist in scores.items():
            line = name + " " + " ".join(map(str, sublist))
            file.write(line + "\n")
    return path
        
def save_scores(scores, epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["score"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        for name, sublist in scores.items():
            line = name + " " + " ".join(map(str, sublist))
            file.write(line + "\n")
    return path

def save_perception_recall(recalls,epoch,trajectory,is_baseline=False):
    path = fill_template(
        path_templates["percep_recall"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        line = ' '.join(f"{num:.1f}" for num in recalls)
        # logger.info(line)
        file.write(line + "\n")
    return path

def load_perception_recall(epoch,trajectory,is_baseline=False):
    path = fill_template(
        path_templates["percep_recall"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "r") as file:
        line = file.readline().strip() 
    recalls_str = line.split()
    recalls = [float(num) for num in recalls_str]
    return recalls

def save_perception_reward(rewards,epoch,trajectory,step, is_baseline=False):
    path = fill_template(
        path_templates["percep_reward"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "a") as file:
        line = str(step)+' '+' '.join(f"{num:.1f}" for num in rewards)
        # logger.info(line)
        file.write(line + "\n")
    return path
def load_rb_reward(epoch,trajectory,is_baseline=False):
    path = fill_template(
        path_templates["rb_reward"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    rewards = []
    with open(path,"r") as file:
        lines = file.readlines()
    agg_line = None
    start = False
    for line in lines:
        if line.startswith('agg'):
            agg_line = line
            start = True
            continue
        if start:
            agg_line = agg_line+line

    agg_line = agg_line.replace('agg', '').strip()
    # Split the line by '][' to separate lists
    raw_lists = agg_line.split('] [')
    # Add missing brackets and convert to float lists
    for raw_list in raw_lists:
        # Remove any leading or trailing brackets and whitespace
        raw_list = raw_list.strip('[] ')
        # Convert string numbers to float
        float_list = [float(x) for x in raw_list.split()]
        rewards.append(float_list)
    return rewards


def load_perception_reward(epoch,trajectory,is_baseline=False):
    path = fill_template(
        path_templates["percep_reward"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    rewards = []
    with open(path, "r") as file:
        for line in file:
            parts = line.split()
            rewards.append([float(x) for x in parts[1:]])
    # max_length = max(len(row) for row in rewards)
    # padded_data = [row + [0] * (max_length - len(row)) for row in rewards]
    # rewards = np.array(padded_data)
    return rewards


def load_scores(epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["score"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    scores = {}
    with open(path, "r") as file:
        for line in file.readlines():
            tokens = line.strip().split()
            name = tokens[0]
            string_numbers = tokens[1:]
            scores_list = list(map(float, string_numbers))
            scores[name] = scores_list
    return scores

def loadd_scores_boxwise(epoch,trajectory, is_baseline=False):
    path = fill_template(
        path_templates["score"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    scores = {}
    with open(path, "r") as file:
        for line in file.readlines():
            tokens = line.strip().split()
            name = tokens[0]
            string_numbers = tokens[1:]
            scores_list = list(map(float, string_numbers))
            scores[name] = scores_list
    return scores


def save_ego_image(image, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["ego_image"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return save_image(image, path)


def load_ego_image(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["ego_image"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_image(path)


def save_birdseye_image(image, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["birdseye_image"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return save_image(image, path)


def load_birdseye_image(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["birdseye_image"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_image(path)


def save_depth_image_raw(image, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["depth_image_raw"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    image.convert(cc.Raw)
    image.save_to_disk(path)
    return path


def load_depth_image_raw(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["depth_image_raw"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_image(path)


def save_depth_image_log(image, epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["depth_image_log"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    image.convert(cc.LogarithmicDepth)
    image.save_to_disk(path)
    return path


def load_depth_image_log(epoch, trajectory, step, is_baseline=False):
    path = fill_template(
        path_templates["depth_image_log"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
        step=step,
    )
    return load_image(path)


def save_vehicle_states(vehicle_states, epoch, trajectory, is_baseline=False):
    speeds, locations = vehicle_states
    path = fill_template(
        path_templates["vehicle_state"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        for speed, location in zip(speeds, locations):
            file.write(f"{speed} {location.x} {location.y} {location.z}\n")
    return path


def load_vehicle_states(epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["vehicle_state"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    speeds = []
    locations = []
    with open(path, "r") as file:
        for line in file.readlines():
            speed, x, y, z = line.strip().split(" ")
            speeds.append(float(speed))
            locations.append((float(x), float(y), float(z)))
    return speeds, locations


def save_vehicle_controls(vehicle_controls, epoch, trajectory, is_baseline=False):
    throttle_list = []
    steer_list = []
    brake_list = []
    hand_brake_list = []
    reverse_list = []
    manual_gear_shift_list = []
    gear_list = []
    for control in vehicle_controls:
        throttle_list.append(control.throttle)
        steer_list.append(control.steer)
        brake_list.append(control.brake)
        hand_brake_list.append(control.hand_brake)
        reverse_list.append(control.reverse)
        manual_gear_shift_list.append(control.manual_gear_shift)
        gear_list.append(control.gear)

    path = fill_template(
        path_templates["vehicle_control"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        file.write(" ".join(map(str, throttle_list)) + "\n")
        file.write(" ".join(map(str, steer_list)) + "\n")
        file.write(" ".join(map(str, brake_list)) + "\n")
        file.write(" ".join(map(str, hand_brake_list)) + "\n")
        file.write(" ".join(map(str, reverse_list)) + "\n")
        file.write(" ".join(map(str, manual_gear_shift_list)) + "\n")
        file.write(" ".join(map(str, gear_list)) + "\n")
    return path


def load_vehicle_controls(epoch, trajectory, is_baseline=False):
    path = fill_template(
        path_templates["vehicle_control"],
        experiment_name=cfg.experiment_name,
        sample_or_baseline=sample_or_baseline(is_baseline),
        epoch=epoch,
        trajectory=trajectory,
    )
    with open(path, "r") as file:
        lines = file.readlines()
        throttle_list = list(map(float, lines[0].strip().split(" ")))
        steer_list = list(map(float, lines[1].strip().split(" ")))
        brake_list = list(map(float, lines[2].strip().split(" ")))
        hand_brake_list = list(map(bool, lines[3].strip().split(" ")))
        reverse_list = list(map(bool, lines[4].strip().split(" ")))
        manual_gear_shift_list = list(map(bool, lines[5].strip().split(" ")))
        gear_list = list(map(int, lines[6].strip().split(" ")))
    return (
        throttle_list,
        steer_list,
        brake_list,
        hand_brake_list,
        reverse_list,
        manual_gear_shift_list,
        gear_list,
    )


def save_checkpoint(model, epoch):
    checkpoint_path = os.path.join(
        "/result",
        cfg.experiment_name,
        "checkpoints",
        f"{epoch}.pth",
    )
    make_dir(os.path.dirname(checkpoint_path))
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def get_checkpoint_path(epoch):
    checkpoint_path = os.path.join(
        "/result",
        cfg.experiment_name,
        "checkpoints",
        f"{epoch}.pth",
    )
    return checkpoint_path


def save_metrics(metrics, epoch):
    path = fill_template(
        path_templates["metrics"],
        experiment_name=cfg.experiment_name,
        epoch=epoch,
    )
    dir = os.path.dirname(path)
    make_dir(dir)
    with open(path, "w") as file:
        file.write(metrics)
    return path

def calculate_iou(bbox1, bbox2):
    # Calculate intersection coordinates
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def compute_recall(bboxes, bboxes_gt, labels,labels_gt,iou_threshold=0.5):
    true_positives = 0
    false_negatives = 0
    recall = 0
    for bbox,label in zip(bboxes,labels):
        bbox_is_tp = False
        for bbox_gt,label_gt in zip(bboxes_gt,labels_gt):
            if label != label_gt:
                continue  # Labels don't match, skip
            iou = calculate_iou(bbox, bbox_gt)
            if iou >= iou_threshold:
                bbox_is_tp = True
                break
        if bbox_is_tp:
            true_positives += 1
        else:
            false_negatives += 1
    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    return recall

def trajectory_recall(epoch,trajectory,is_baseline = False):
    recalls = []
    for step in range(cfg.train.trajectory_length):
        objects = load_predicted_objects(epoch,trajectory,step,is_baseline = is_baseline)
        objects_gt = load_ground_truth_objects(epoch,trajectory,step,is_baseline = is_baseline)
        bboxes = [object.bbox_xyxy for object in objects]
        bboxes_gt = []
        for object in objects_gt:
            bbox_gt = object.bbox_xyxy
            if bbox_gt[0] > bbox_gt[2] or bbox_gt[1] > bbox_gt[3]:
                continue
            if bbox_gt[0] > cfg.carla.ego_camera.image_width:
                continue
            if bbox_gt[1] > cfg.carla.ego_camera.image_height:
                continue
            if bbox_gt[2] < 1:
                continue
            if bbox_gt[3] < 1:
                continue            
            if bbox_gt[0] < 1:
                bbox_gt[0] = 1
            if bbox_gt[1] < 1:
                bbox_gt[1] = 1
            if bbox_gt[2] > cfg.carla.ego_camera.image_width:
                bbox_gt[2] = cfg.carla.ego_camera.image_width
            if bbox_gt[3] > cfg.carla.ego_camera.image_height:
                bbox_gt[3] = cfg.carla.ego_camera.image_height
            bboxes_gt.append(bbox_gt)
        # logger.info(f"bboxes are {bboxes}")
        # logger.info(f"bboxes_gt are {bboxes_gt}")
        categories = [object.category for object in objects]
        categories_gt = [object.category for object in objects_gt]
        recall = compute_recall(bboxes,bboxes_gt,categories,categories_gt,iou_threshold=0.5)
        recalls.append(recall)
    return recalls