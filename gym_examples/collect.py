from scenic.gym.env import CarlaEnv
import scenic
from vocabulary import Vocabulary
import pygame
import utils
from rulebook import RuleBook
from agent_behaviors import BehaviorAgent
from models import PretrainedImageCaptioningModel, resize_bboxes_xyxy, get_train_transform, predict_detections, get_bboxes_and_labels, bbox_resize_xxyy, compute_matchness, compute_recall
from config import cfg
from object_info import ObjectInfo
import torch
import file as futils
from object_annotator import auto_annotate
import numpy as np
from agents.tools.misc import get_speed

def collect_scene(model,
    device,
    epoch,
    scenario,
    scene_length,
    vocabulary,
    seed,
    env,
    episode,
    render = False, is_baseline = False):
    
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
    
    vehicle_speeds = []
    vehicle_locations = []
    total_detections = 0
    control_history = []
    total_recall = []
    image_w = int(env.simulation.ego_camera.sensor.attributes["image_size_x"])
    image_h = int(env.simulation.ego_camera.sensor.attributes["image_size_y"])
    
    
    
    for step in range(scene_length):
        model.eval()
        
        vehicle_speeds.append(get_speed(ego))
        vehicle_locations.append(ego.get_location())
        
        ego_image = obs
        ego_image_array = utils.ego_image_array(ego_image)
        #ego_image.save("images/ego_image%d.jpg" % step)
        
        #ego_image.save_to_disk("images/ego_image%d.jpg" % step)
        #Saving ego image
        futils.save_ego_image(ego_image, epoch, episode, step, is_baseline=is_baseline)
        
        #Not saving birdseye image for now
        
        depth_image = env.getDepthImage()
        depth_image_array = utils.image_to_grayscale_depth_array(depth_image)
        
        
        transformed_ego_image = ego_image_transform(ego_image_array)
        device_ego_image = transformed_ego_image.to(device)
    
        predicted_detections, predicted_captions, predicted_indices = predict_detections(
                model, device, device_ego_image.unsqueeze(0), vocabulary
        )
        
        futils.save_predicted_captions(predicted_captions, epoch, episode, step, is_baseline=is_baseline)
        

        
        
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
        
        futils.save_depth_image_raw(depth_image, epoch, episode, step, is_baseline=is_baseline)
        futils.save_depth_image_log(depth_image, epoch, episode, step, is_baseline=is_baseline)
        
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
        
        futils.save_predicted_objects(predicted_objects, epoch, episode, step, is_baseline=is_baseline)
        
        ground_truth_objects = utils.get_ground_truth_objects(
            ego,
            env.simulation.ego_camera.sensor,
        )
        
        futils.save_ground_truth_objects(ground_truth_objects, epoch, episode, step, is_baseline=is_baseline)
        
        world = env.simulation.client.get_world()
        vehicles = world.get_actors().filter("vehicle.*")
        peds = world.get_actors().filter("walker.pedestrian.*")
        all_actors = []
        
        for v in vehicles:
            all_actors.append(v)
        for p in peds:
            all_actors.append(p)
            
        filtered, removed = auto_annotate(all_actors, env.simulation.depth_camera.sensor, depth_image_array, json_path = cfg.train.json_path)
        
        
        bboxes, labels = get_bboxes_and_labels(predicted_indices[1:],vocabulary.get_idx_to_word())
        agent_rulebook.update_bbox2d(bboxes,labels)
        bboxes_gt = []
        labels_gt = []
        bboxes_annotated = filtered['bbox']
        labels_annotated = filtered['class']
        for i in range(len(labels_annotated)):
            bbox = bbox_resize_xxyy(bboxes_annotated[i])
            if labels_annotated[i] == 0:
                label = 'car'
            else:
                label = 'person'
            bboxes_gt.append(bbox)
            labels_gt.append(label)
        agent_rulebook.update_bbox_and_label_gt(bboxes_gt,labels_gt)
        recall = compute_recall(bboxes,bboxes_gt,labels,labels_gt)
        total_recall.append(recall*100)
        r = compute_matchness(bboxes,bboxes_gt,labels,labels_gt)
        r = np.array(r)
        r_cumsum = np.cumsum(r)
        r_cumsum = np.repeat(r_cumsum,5)

        futils.save_perception_reward(
        r_cumsum,
        epoch,
        episode,
        step,
        is_baseline=is_baseline,
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
        
    futils.save_perception_recall(total_recall, epoch, episode, is_baseline=is_baseline)
    futils.save_vehicle_states((vehicle_speeds, vehicle_locations), epoch, episode, is_baseline=is_baseline)
    futils.save_control_history(control_history, epoch, episode, is_baseline=is_baseline)
    scores = agent_rulebook.get_violation_scores()
    
    if cfg.common.rule_boxwise:
        futils.save_scores_boxwise(scores, epoch, episode, is_baseline=is_baseline)
    else:
        futils.save_scores(scores, epoch, episode, is_baseline=is_baseline)
    
def collect(seed = 0, render = False, num_episodes = 1, scene_length = 100, epoch = 0, is_baseline = False):
    map_path = scenic.syntax.veneer.localPath("~/Scenic/assets/maps/CARLA/Town10HD.xodr")
    carla_map = "Town10HD"
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
    
    for episode in range(num_episodes):
        scenario = scenic.scenarioFromFile(f"test0.scenic", mode2D=True)
        collect_scene(model = model, device = device, epoch = epoch, scenario = scenario, scene_length= scene_length, vocabulary = vocabulary, seed = seed, env = env, episode = episode, render = render, is_baseline = is_baseline)
        seed += 1  


collect(seed = 0, render = False, num_episodes = 3, scene_length = 60, epoch = 0, is_baseline = False)