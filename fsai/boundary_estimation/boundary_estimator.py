from typing import List, Dict, Set

import numpy as np
from scipy.spatial import Delaunay

from fsai.objects.Live import Line
from fsai.objects.cone import Cone, CONE_COLOR_BIG_ORANGE, CONE_COLOR_BLUE, CONE_COLOR_YELLOW


def get_delauny_triangles(self):
    triangles, invalid = [], []

    all_cones, delaunay = self.__get_delauny_map()
    missed_cones = set(all_cones)

    for triangle in delaunay:
        cone_a, cone_b, cone_c = triangle[0], triangle[1], triangle[2]
        triangle_colours = [cone_a.color, cone_b.color, cone_c.color]

        # bool statements to check if a triangle is part of the track
        orange_pair = triangle_colours.count(CONE_COLOR_BIG_ORANGE) == 2
        all_mixed = CONE_COLOR_BIG_ORANGE in triangle_colours and CONE_COLOR_BLUE in triangle_colours and CONE_COLOR_YELLOW in triangle_colours
        two_blue = triangle_colours.count(CONE_COLOR_BLUE) == 2 and CONE_COLOR_YELLOW in triangle_colours
        two_yellow = triangle_colours.count(CONE_COLOR_YELLOW) == 2 and CONE_COLOR_BLUE in triangle_colours

        # if triangle meets criteria then add add to valid triangles
        if orange_pair or all_mixed or two_blue or two_yellow:
            triangles.append(triangle)
            if cone_a in missed_cones: missed_cones.remove(cone_a)
            if cone_b in missed_cones: missed_cones.remove(cone_b)
            if cone_c in missed_cones: missed_cones.remove(cone_c)
        else:
            invalid.append(triangle)

    return triangles


def create_boundary(self):
    triangles = self.create_delaunay_graph()

    blue_boundaries = []
    yellow_boundaries = []
    orange_boundaries = []

    track_graph: Dict[Cone: Set[Cone]] = {}
    for triangle in triangles:
        for i in range(len(triangle)):
            if triangle[i] not in track_graph:
                track_graph[triangle[i]] = set()
            track_graph[triangle[i]].add(triangle[i - 1])
            track_graph[triangle[i]].add(triangle[i - 2])

    for cone in track_graph:
        if cone.color == CONE_COLOR_BLUE:
            for connection in track_graph[cone]:
                if connection.color == CONE_COLOR_BLUE:
                    blue_boundaries.append(Line(cone.point, connection.point))

        if cone.color == CONE_COLOR_YELLOW:
            for connection in track_graph[cone]:
                if connection.color == CONE_COLOR_YELLOW:
                    yellow_boundaries.append(Line(cone.point, connection.point))

        if cone.color == CONE_COLOR_BIG_ORANGE:
            cones = [{
                "cone": connection,
                "length": connection.point.distance(cone.point)
            } for connection in track_graph[cone]]
            cones = sorted(cones, key=lambda item: item['length'])
            closest_blue, closest_yellow = [], []
            for closest_cone in cones:
                if closest_cone["cone"].color == CONE_COLOR_YELLOW: closest_yellow.append(closest_cone["cone"])
                if closest_cone["cone"].color == CONE_COLOR_BLUE: closest_blue.append(closest_cone["cone"])

                if len(closest_blue) >= 2:
                    orange_boundaries.append(Line(closest_blue[0].point, cone.point))
                    orange_boundaries.append(Line(closest_blue[1].point, cone.point))
                    break
                if len(closest_yellow) >= 2:
                    orange_boundaries.append(Line(closest_yellow[0].point, cone.point))
                    orange_boundaries.append(Line(closest_yellow[1].point, cone.point))
                    break

    return blue_boundaries, yellow_boundaries, orange_boundaries


def __get_delauny_map(
        blue_cones: List[Cone],
        yellow_cones: List[Cone],
        orange_cones: List[Cone],
        big_orange_cones: List[Cone]
):
    big_orange_cones = __merge_big_orange_cones(big_orange_cones)
    all_cones = blue_cones + yellow_cones + big_orange_cones + orange_cones

    delaunay = Delaunay(np.asarray([[cone.point.x, cone.point.y] for cone in all_cones]))
    # convert triangles from list of indexes to list of cones
    delaunay = [[all_cones[tri[0]], all_cones[tri[1]], all_cones[tri[2]]] for tri in delaunay.simplices]

    return all_cones, delaunay

def __merge_big_orange_cones(orange_cones):
    merged_orange_cones: List[Cone] = []
    for cone in orange_cones:
        too_close = False
        for merged_cone in merged_orange_cones:
            too_close = too_close or cone.point.distance(merged_cone.point) < 2.75
        if not too_close:
            merged_orange_cones.append(cone)
    return merged_orange_cones
