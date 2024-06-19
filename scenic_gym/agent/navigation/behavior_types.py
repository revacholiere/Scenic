# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """


class Cautious(object):
    """Class for Cautious agent."""
    max_speed = 40
    speed_lim_dist = 6
    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 6
    tailgate_counter = 0


class Normal(object):
    """Class for Normal agent."""
    max_speed = 90
    speed_lim_dist = 0
    speed_decrease = 20
    safety_time = 3
    min_proximity_threshold = 10
    braking_distance = 5
    tailgate_counter = 0


class Aggressive(object):
    """Class for Aggressive agent."""
    max_speed = 150
    speed_lim_dist = -50
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 8
    braking_distance = 3
    tailgate_counter = -1