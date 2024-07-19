from carla_env import CarlaEnv
from ultralytics import YOLO
import pygame
import random
import scenic
from behavior_agent import BehaviorAgent
from rulebook import RuleBook
from object_info import create_obj_list
from utils import (
    image_to_array,
    image_to_grayscale_depth_array,
    draw_boxes,
    get_ground_truth_bboxes,
    image_to_surface,
    pil_to_surface,
)
from PIL import Image


def main(seed, num_episodes, scene_length, timestep, render=False, save=False):  # Test the environment
    map_path = scenic.syntax.veneer.localPath("~/Scenic/assets/maps/CARLA/Town10HD.xodr")
    carla_map = "Town10HD"
    env = CarlaEnv(carla_map=carla_map, map_path=map_path, render=False, timestep=timestep)
    model = YOLO("./yolov5su.pt")

    # scenario = scenic.scenarioFromFile("test.scenic", mode2D=True)
    if render:
        pygame.init()
        display = pygame.display.set_mode(
            (1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

    for j in range(1, num_episodes+1):

        #scenario = scenic.scenarioFromFile(f"Carla_Challenge/carlaChallenge{j}.scenic", mode2D=True)
        scenario = scenic.scenarioFromFile(f"test0.scenic", mode2D=True)
        random.seed(seed + j)
        try: scene, _ = scenario.generate()
        except: continue #   scenic may fail to generate scenario
        env.set_scene(scene)
        obs = env.reset()
        agent = BehaviorAgent(env.getEgo(), behavior="normal")
        rulebook = RuleBook(env.getEgo())
        obj_list = []
        agent.update_object_information(obj_list)

        for i in range(int(scene_length/timestep)):  # number of timesteps

            obs_array = image_to_array(obs)
            depth_image = env.getDepthImage()
            depth_array = image_to_grayscale_depth_array(depth_image)

            pred = model(obs_array, verbose=False, classes=[1, 2, 3, 5, 7], conf=0.7)
            pred_np = pred[0].cpu().numpy()
            img = Image.fromarray(pred[0].plot())
            bboxes = get_ground_truth_bboxes(
                env.simulation.world.get_actors().filter("*vehicle*"),
                env.getEgo(),
                env.simulation.ego_camera.sensor,
            )
            if len(bboxes) > 0:
                draw_boxes(img, bboxes[:, 1:][:, :4], (255, 0, 0, 100))

            obj_list = (
                create_obj_list(env.simulation, pred_np.boxes, depth_array)
                if len(pred[0].boxes) > 0
                else []
            )

            # print(pred)
            agent.update_object_information(obj_list)

            rulebook.step()

            ctrl = agent.run_step()

            obs, _, __, ___ = env.step(ctrl)

            if i % 1 == 0:
                # img.save_to_disk("seed%d_video%d/%.6d.jpg" % (seed, j, obs.frame))
                # img.save("images/seed%d_video%d%.6d.jpg" % (seed, j, obs.frame))
                # print(type(env.simulation.ego_camera._surface))
                # print(type(obs), type(img))
                surf_raw = image_to_surface(obs)
                surf_boxes = pil_to_surface(img)

                # pygame.image.save(surf_raw, "images/surf_raw%d.jpg" % (obs.frame))
                # pygame.image.save(surf_boxes, "images/surf_boxes%d.jpg" % (obs.frame))
                if render:
                    display.blit(surf_boxes, (0, 0))
                    pygame.display.flip()
                if save:
                    pygame.image.save(surf_boxes, "images/surf_boxes%d.jpg" % (obs.frame))

        print(f"end episode {j}")
        env.end_episode()

    env.close()


main(seed=1, num_episodes=9, scene_length=10, timestep=0.1, render=False, save=True)
