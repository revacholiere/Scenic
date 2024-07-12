from carla_env import CarlaEnv
import scenic
from vocabulary import Vocabulary
import pygame
import utils
from rulebook import RuleBook
from agent_behaviors import BehaviorAgent
from models import PretrainedImageCaptioningModel, resize_bboxes_xyxy, get_train_transform, predict_detections
from config import cfg
from object_info import ObjectInfo
import torch


def collect_scene(model,
    device,
    epoch,
    scenario,
    scene_length,
    vocabulary,
    seed,
    env,
    render = False):
    
    if render:
        pygame.init()
        display = pygame.display.set_mode(
            (1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
    
    
    
    utils.set_seeds(seed)
    scene, _ = scenario.generate()
    env.set_scene(scene)
    obs = env.reset()
    
    ego = env.getEgo()
    
    
    agent_behavior_control = BehaviorAgent(ego)
    agent_rulebook = RuleBook(ego)
    
    ego_image_transform = get_train_transform()
    
    
    total_detections = 0
    control_history = []
    
    image_w = int(env.simulation.ego_camera.sensor.attributes["image_size_x"])
    image_h = int(env.simulation.ego_camera.sensor.attributes["image_size_y"])
    
    
    
    for step in range(scene_length):
        model.eval()
        ego_image = obs
        ego_image_array = utils.ego_image_array(ego_image)
        #ego_image.save("images/ego_image%d.jpg" % step)
        ego_image.save_to_disk("images/ego_image%d.jpg" % step)
        depth_image = env.getDepthImage()
        depth_image_array = utils.image_to_grayscale_depth_array(depth_image)
        
        
        transformed_ego_image = ego_image_transform(ego_image_array)
        device_ego_image = transformed_ego_image.to(device)
    
        predicted_detections, predicted_captions = predict_detections(
                model, device, device_ego_image.unsqueeze(0), vocabulary
        )
        bboxes_xyxy, categories, confidences = predicted_detections
        total_detections += len(categories)
        bboxes_xyxy = resize_bboxes_xyxy(
            bboxes_xyxy,
            (cfg.common.img_width_resize, cfg.common.img_height_resize),
            (
                image_w,
                image_h,
            ),
        )

        locations, bboxes_3d = utils.compute_locations_and_bboxes_3d(
                depth_image_array,
                bboxes_xyxy,
                env.simulation.depth_camera.sensor,
            )
        
        distances = [ego.get_location().distance(l) for l in locations]
        speeds = []
        for i in range(len(bboxes_3d)):
            if categories[i] in vocabulary.pedestrian_tokens:
                speeds.append(0)
            else:
                speeds.append(10)
                
        predicted_objects = [
            ObjectInfo(
                bbox_xyxy, category, confidence, location, distance, speed, bbox_3d
            )
            for bbox_xyxy, category, confidence, location, distance, speed, bbox_3d in zip(
                bboxes_xyxy,
                categories,
                confidences,
                locations,
                distances,
                speeds,
                bboxes_3d,
            )
        ]
        
        ground_truth_objects = utils.get_ground_truth_objects(
            ego,
            env.simulation.ego_camera.sensor,
        )
        
        for obj in predicted_objects:
            if obj.category in vocabulary.pedestrian_tokens:
                obj.category = 0
            elif obj.category in vocabulary.vehicle_tokens:
                obj.category = 1
        agent_behavior_control.update_object_information(predicted_objects)
        control = agent_behavior_control.run_step()
        control_history.append(control) 
        
        if render:
            img = utils.image_to_surface(ego_image)
            display.blit(img, (0, 0))
            pygame.display.flip()
        
        
        obs, _, __, ___ = env.step(control)
        
        agent_rulebook.step()
        
    env.end_episode()
        
    
    
    
    
    
def collect(seed = 0, render = False):
    map_path = scenic.syntax.veneer.localPath("~/Scenic/assets/maps/CARLA/Town01.xodr")
    carla_map = "Town01"
    env = CarlaEnv(carla_map=carla_map, map_path=map_path, render=False)
    vocabulary = Vocabulary()
    num_categories = len(vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PretrainedImageCaptioningModel(
        cfg.common.hidden_size,
        cfg.common.num_heads,
        cfg.common.num_encoder_layers,
        cfg.common.num_decoder_layers,
        num_categories,
        cfg.common.image_size,
        cfg.common.patch_size,
        cfg.common.max_caption_length,
    ).to(device)
    
    
    scenario = scenic.scenarioFromFile(f"test1.scenic", mode2D=True)
    collect_scene(model, device, 0, scenario, 100, vocabulary, seed, env, render)

collect(0, False)