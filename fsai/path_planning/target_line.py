from typing import List

from fsai.objects.point import Point
from fsai.objects.waypoint import Waypoint


def get_points_from_waypoints(waypoints: List[Waypoint]):
    points: List[Point] = []
    for waypoint in waypoints:
        points.append(waypoint.get_optimum_point())
    return points
