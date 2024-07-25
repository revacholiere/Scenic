import logging
import math
import queue
import random
import sys
import time

import carla
import cv2
import numpy as np
from agents.tools.misc import get_speed
from shapely.geometry import Polygon
from shapely import distance
from utils import project_points_world_to_image
from config import cfg
from models import calculate_iou

logger = logging.getLogger(__name__)


class Rule:
    def __init__(
        self,
        name,
        type,
        value,
        weight,
        level,
        actor=None,
        min=0,
        max=math.inf,
        angle=360,
        norm=1,
        a_brake_min = 0,
        a_brake_max = 0,
        a_max = 10,
        rho = 0.1,
    ):
        self.name = name
        self.type = type
        self.value = value
        self.weight = weight
        self.level = level
        self.actor = actor
        self.min = min
        self.max = max
        self.angle = angle
        if cfg.common.exp_norm:
            self.norm = 1
        else:
            self.norm = norm
        self.a_brake_min = a_brake_min
        self.a_brake_max = a_brake_max
        self.a_max = a_max
        self.rho = rho

    def __str__(self):
        return f"Rule({self.name}, {self.type}, {self.value}, {self.weight}, {self.level}, {self.actor}, {self.min}, {self.max}, {self.angle}, {self.norm}, {self.a_brake_max}, {self.a_brake_min}, {self.a_max}, {self.rho})"


class RuleBook(object):
    def __init__(self, vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._trajectory_log = []
        self._acceleration_log = []
        self._velocity_log = []
        self._velocity3D_log = []
        self._transform_log = []
        self._bbox_log = []
        self._control_log = []
        self._bbox2d_log = []
        self._label_log = []
        self._rules = []
        self._scores = {}
        self._rules = RuleBook._build_rules()
        self._ego_cam = None
        self._bboxes_gt = []
        self._labels_gt = []
        for rule in self._rules:
            self._scores[rule.name] = []
            self._scores[rule.name+"raw"] = []
            logger.debug(f"Rule {rule} added to rulebook")
        self._scores["agg"] = []
    
    def set_ego_cam(self,cam):
        self._ego_cam = cam

    def _distance(self, actor):
        return actor.get_location().distance(self._vehicle.get_transform().location)
    
    def update_bbox2d(self,bbox2Ds,labels):
        self._bbox2d_log = bbox2Ds
        self._label_log = labels

    def update_bbox_and_label_gt(self,bbox2Ds_gt,labels_gt):
        self._bboxes_gt = bbox2Ds_gt
        self._labels_gt = labels_gt

    def _within_angle(self, actor, angle):
        forward = self._vehicle.get_transform().get_forward_vector()
        logger.debug(f"forward: {forward}")
        actor_vector = actor.get_location() - self._vehicle.get_transform().location
        actor_vector = actor_vector / actor_vector.length()
        logger.debug(f"actor_vector: {actor_vector}")
        dot = forward.dot(actor_vector)
        cos = math.cos(angle * math.pi / 180)
        logger.debug(f"dot: {dot}, cos: {cos}")
        return dot > cos

    def _normalize(self, score, norm = 1):
        if norm == 1:
            return 1 - math.exp(-score)
        else:
            return score/norm 
    

    def _detect_collision(self, target):
        target_bb = target.bounding_box
        target_vertices = target_bb.get_world_vertices(target.get_transform())
        target_list = [[v.x, v.y, v.z] for v in target_vertices]
        target_polygon = Polygon(target_list)

        self_vertices = self._bbox_log[-1].get_world_vertices(
            self._transform_log[-1]
        )

        self_vertices_pre = self._bbox_log[-2].get_world_vertices(
            self._transform_log[-2]
        )
        self_list = [[v.x, v.y, v.z] for v in self_vertices+self_vertices_pre]
        # for v in self_vertices:
        #     self_list.append([v.x, v.y,v.z])

        self_polygon = Polygon(self_list)
        # if distance(self_polygon,target_polygon) < 0.1:
        #     logger.info(f"two actor within distance{distance(self_polygon,target_polygon)}")
        #     logger.info(f"intersection info is {self_polygon.intersects(target_polygon)}")
        #     logger.info(f"self location is {self._vehicle.get_transform()}, actor location is {target.get_transform()}")
        #     logger.info(f"self location is {self._transform_log[-1]}")

        #     return True

        return self_polygon.intersects(target_polygon)

    def step(self):
        self._trajectory_log.append(self._vehicle.get_location())
        self._velocity_log.append(get_speed(self._vehicle))
        self._transform_log.append(self._vehicle.get_transform())
        self._acceleration_log.append(self._vehicle.get_acceleration())
        self._control_log.append(self._vehicle.get_control())
        self._bbox_log.append(self._vehicle.bounding_box)
        self._velocity3D_log.append(self._vehicle.get_velocity())
        if cfg.common.rule_boxwise:
            agg_score = np.zeros(0)
            for rule in self._rules:
                if rule.type == "collision-boxwise":
                    score_raw = self._collision_rule_boxwise(rule)
                elif rule.type == "rss-boxwise":
                    score_raw = self._rss_rule_boxwise(rule)
                elif rule.type == "hard-brake-boxwise":
                    score_raw = self._hard_brake_rule_boxwise(rule)
                else:
                    logger.warning(f"Unknown rule type: {rule.type}")
                self._scores[rule.name+"raw"].append(score_raw)
                max_len = max(len(agg_score), len(score_raw))
                score_raw = np.pad(score_raw,(0,max_len - len(score_raw)),'constant')
                agg_score = np.pad(agg_score,(0,max_len - len(agg_score)),'constant')
                agg_score += score_raw
            self._scores["agg"].append(agg_score)
        else:
            agg_score = 0
            for rule in self._rules:
                if rule.type == "collision":
                    score_raw = self._collision_rule(rule)
                    score = self._normalize(score_raw,rule.norm)
                elif rule.type == "rss":
                    score_raw = self._rss_rule(rule)
                    score = self._normalize(score_raw,rule.norm)
                elif rule.type == "proximity":
                    score_raw = self._proximity_rule(rule)
                    score = self._normalize(score_raw,rule.norm)
                elif rule.type == "velocity":
                    score_raw = self._velocity_rule(rule)
                    score = self._normalize(score_raw,rule.norm)
                elif rule.type == "distance":
                    score_raw = self._distance_rule(rule)
                    score = self._normalize(score_raw,rule.norm)
                elif rule.type == "hard_brake":
                    score_raw = self._hard_brake_rule(rule)
                    score = self._normalize(score_raw,rule.norm)
                elif rule.type == "smooth":
                    score_raw = self._smooth_run_rule(rule)
                    score = self._normalize(score_raw,rule.norm)
                else:
                    logger.warning(f"Unknown rule type: {rule.type}")
                self._scores[rule.name+"raw"].append(score_raw)
                self._scores[rule.name].append(score)
                agg_score += 2 ** (1 - rule.level) * score * rule.weight
            self._scores["agg"].append(agg_score)

    def _hard_brake_rule(self,rule):
        threshold = 5
        score = 0
        last_10_control = []
        last_acc = self._acceleration_log[-1]
        last_vec = self._velocity_log[-1]

        if last_acc.dot(last_vec) < 0 and last_acc.x^2+last_acc.y^2+last_acc.z^2 > 4:
            score = rule.value*(last_acc.x^2+last_acc.y^2+last_acc.z^2 - threshold)   

        if len(self._control_log) <=10:
            last_10_control = self._control_log
        else:
            last_10_control = self._control_log[-10:]

        hard_brake_count = 0
        for control in last_10_control:
            if control.brake > 0.5:
                hard_brake_count+=1
        
        if hard_brake_count >= threshold:
            score =  rule.value*(hard_brake_count - threshold)   
        return score
    
    def _smooth_run_rule(self,rule):
        score = 0
        last_10_control = []
        threshold = 0.2
        if len(self._control_log) <=10:
            last_10_control = self._control_log
        else:
            last_10_control = self._control_log[-10:]
        total_rate_of_change = 0
        for i in range(1, len(last_10_control)):
            current_ctrl = last_10_control[i].throttle - last_10_control[i].brake
            last_ctrl = last_10_control[i-1].throttle - last_10_control[i-1].brake
            total_rate_of_change = abs(current_ctrl-last_ctrl)
        avg_rate_of_change = total_rate_of_change/len(last_10_control)
        score = avg_rate_of_change*rule.value
        return score

    def _rss_rule(self,rule):
        score = 0
        actor_list = self._world.get_actors().filter(rule.actor)
        for actor in actor_list:
            if actor.id == self._vehicle.id:
                continue
            if self._within_angle(actor, rule.angle) and self._within_same_lane(actor):
                if self._detect_rss(actor,rule):
                    score += score + rule.value
        return score
    
    def _detect_rss(self,target,rule):
        v_r = self._vehicle.get_velocity().length()
        v_f = target.get_velocity().length()
        a_brake_min = rule.a_brake_min
        a_brake_max = rule.a_brake_max
        a_max = rule.a_max
        rho = rule.rho
        d = v_r*rho + 0.5*a_max*rho*rho+(v_r+a_max*rho)*(v_r+a_max*rho)/(2*a_brake_min)-v_f*v_f/(2*a_brake_max)
        location_r = self._vehicle.get_location()
        location_f = target.get_location()
        distance =  location_f.distance(location_r)
        if distance > max(0,d):
            return False
        return True
        
    def _within_same_lane(self,target):
        map = self._world.get_map()
        wp_self = map.get_waypoint(self._vehicle.get_location())
        wp_target = map.get_waypoint(target.get_location())
        if wp_self.lane_id == wp_target.lane_id:
            return True
        return False

    def _collision_rule(self, rule):
        score = 0
        if len(self._velocity_log) < 2:
            return score

        current_velocity = self._velocity_log[-1]
        actor_list = self._world.get_actors().filter(rule.actor)
        for actor in actor_list:
            if actor.id == self._vehicle.id:
                continue
            if self._detect_collision(actor):
                logger.debug(f"Collision detected with {actor.id}")
                score += score + rule.value * current_velocity
        return score
    
    def _distance_rule(self,rule):
        score = 0
        if len(self._trajectory_log) < 2:
               return score
        current_location = self._trajectory_log[-1]
        previous_location = self._trajectory_log[-2]
        score = 1 - current_location.distance(previous_location)
        if score < 0: 
            score = 0
        return score

    def _proximity_rule(self, rule):
        score = 0
        if len(self._velocity_log) < 2:
            return score

        current_velocity = self._velocity_log[-1]
        previous_velocity = self._velocity_log[-2]
        acceleration = current_velocity - previous_velocity

        actor_list = self._world.get_actors().filter(rule.actor)
        for actor in actor_list:
            if actor.id == self._vehicle.id:
                continue
            if self._distance(actor) < rule.min or self._distance(actor) > rule.max:
                continue
            if self._within_angle(actor, rule.angle) and acceleration < 0:
                score += score + rule.value * current_velocity
        logger.debug(f"Proximity score[{rule.name}]: {score}")
        return score

    def _velocity_rule(self, rule):
        score = 0
        if len(self._velocity_log) < 2:
            return score

        current_velocity = self._velocity_log[-1]
        previous_velocity = self._velocity_log[-2]
        acceleration = current_velocity - previous_velocity

        if current_velocity < rule.min:
            score += score + rule.value * current_velocity
        elif current_velocity > rule.max:
            score += score + rule.value * current_velocity
        logger.debug(f"Velocity score[{rule.name}]: {score}")
        return score

    def get_violation_scores(self):
        return self._scores
    
    def _rss_rule_boxwise(self,rule):
        bboxes = self._bbox2d_log
        score = np.zeros(len(bboxes)*5)
        actor_list = self._world.get_actors().filter(rule.actor)
        current_velocity = self._velocity_log[-1]
        for actor in actor_list:
            if actor.id == self._vehicle.id:
                continue
            if self._within_angle(actor, rule.angle) and self._within_same_lane(actor):
                if self._detect_rss(actor,rule):
                    indices, labels_check = self._find_bbox_index(actor,"car")
                    if not indices:
                       score = np.concatenate((score,np.ones(5)*(-current_velocity)))
                    else:
                        for index,label in zip(indices,labels_check):
                           score[index*5:index*5+4] = 1
                           if label == True:
                               score[(index+1)*5] = 1
                           else:
                               score[(index+1)*5] = -1                 
        return score
    
    def _hard_brake_rule_boxwise(self, rule):
        # to be fixed 
        bboxes = self._bbox2d_log
        labels = self._label_log
        bboxes_gt = self._bboxes_gt
        labels_gt = self._labels_gt
        score = np.zeros(len(bboxes)*5)
        actor_list = self._world.get_actors().filter(rule.actor)
        last_acc = self._acceleration_log[-1]
        last_acc_value = math.sqrt(last_acc.x ** 2+last_acc.y ** 2+last_acc.z ** 2)
        last_vec = self._velocity3D_log[-1]
        if last_acc.dot(last_vec) < 0 and  last_acc_value > 4:
            index = 0
            # check if the bbox match any of the ground truth
            for bbox,label in zip(bboxes,labels):
                match = False
                for bbox_gt,label_gt in zip(bboxes_gt,labels_gt):
                    iou = calculate_iou(bbox,bbox_gt)
                    if iou > cfg.common.iou_threshold:
                        if label == label_gt:
                            match = True
                            break
                    else:
                        continue
                if not match:
                    score[index*5:index*5+5] = -1*last_acc_value
                index+=1
        return score
        
    def _collision_rule_boxwise(self, rule):
        bboxes = self._bbox2d_log
        score = np.zeros(len(bboxes)*5)
        if len(self._velocity_log) < 2:
            return score

        current_velocity = self._velocity_log[-1]
        actor_list = self._world.get_actors().filter(rule.actor)
        for actor in actor_list:
            if actor.id == self._vehicle.id:
                continue
            if self._detect_collision(actor):
                    indices, labels_check = self._find_bbox_index(actor,"car")
                    if not indices:
                        score = np.concatenate((score,np.ones(5)*(-current_velocity)))
                    else:
                        for index,label in zip(indices,labels_check):
                           score[index*5:index*5+4] = 1
                           if label == True:
                               score[(index+1)*5] = 1
                           else:
                               score[(index+1)*5] = -1
        return score

    def _find_bbox_index(self,actor,label_gt):
        # Find if there is a corresponding bbox with the given bbox gt
        # If there is, then check label
        # If not return empty list
        actor_bbox3D = actor.bounding_box
        world_verts = [
            [v.x, v.y, v.z] for v in actor_bbox3D.get_world_vertices(actor.get_transform())
        ]
        actor_vertices2D = project_points_world_to_image(world_verts, self._ego_cam)        
        x_min = min([p[0] for p in actor_vertices2D])
        y_min = min([p[1] for p in actor_vertices2D])
        x_max = max([p[0] for p in actor_vertices2D])
        y_max = max([p[1] for p in actor_vertices2D])
        actor_bbox2D = [x_min,x_max,y_min,y_max]
        bboxes = self._bbox2d_log
        labels = self._label_log
        count = 0
        index = []
        label_check = []
        for bbox,label in zip(bboxes,labels):
            iou = calculate_iou(bbox,actor_bbox2D)
            if iou > cfg.common.iou_threshold:
                index.append(count)
                if label == label_gt:
                    label_check.append(True)
                else:
                    label_check.append(False)
            count+=1
        return index,label_check
    
    @classmethod
    def get_max_score(cls, force_rebuild=False):
        if force_rebuild or not hasattr(cls, "_rules"):
            rules = cls._build_rules()
        else:
            rules = cls._get_rules()
        max_score = 0
        for rule in rules:
            max_score += 2 ** (1 - rule.level) * rule.weight
        return max_score
    
    @classmethod
    def _build_rules_boxwise(cls):
        rules = []
        for rule in cfg.common.rulebook_boxwise:
            rules.append(
                Rule(
                    name=rule.name,
                    type=rule.type,
                    value=float(rule.value),
                    weight=float(rule.weight),
                    level=int(rule.level),
                    actor=rule.get("actor", None),
                    min=float(rule.get("min", 0)),
                    max=float(rule.get("max", math.inf)),
                    angle=float(rule.get("angle", 360)),
                    norm=float(rule.get("norm",1)),
                    a_brake_max=float(rule.get("a_brake_max",0)),
                    a_brake_min=float(rule.get("a_brake_min",0)),
                    a_max=float(rule.get("a_max",0)),
                    rho=float(rule.get("rho",0))
                )
            )
        RuleBook._cache_rules(rules)
        return rules
    
    @classmethod
    def _build_rules(cls):
        rules = []
        rulebook = None
        if cfg.common.rule_boxwise:
            rulebook = cfg.common.rulebook_boxwise
        else:
            rulebook = cfg.common.rulebook
        for rule in rulebook:
            rules.append(
                Rule(
                    name=rule.name,
                    type=rule.type,
                    value=float(rule.value),
                    weight=float(rule.weight),
                    level=int(rule.level),
                    actor=rule.get("actor", None),
                    min=float(rule.get("min", 0)),
                    max=float(rule.get("max", math.inf)),
                    angle=float(rule.get("angle", 360)),
                    norm=float(rule.get("norm",1)),
                    a_brake_max=float(rule.get("a_brake_max",0)),
                    a_brake_min=float(rule.get("a_brake_min",0)),
                    a_max=float(rule.get("a_max",0)),
                    rho=float(rule.get("rho",0))
                )
            )
        RuleBook._cache_rules(rules)
        return rules

    @classmethod
    def _cache_rules(cls, rules):
        cls._rules = rules

    @classmethod
    def _get_rules(cls):
        return cls._rules
