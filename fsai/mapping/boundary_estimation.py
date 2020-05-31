from typing import List, Dict, Set, Tuple

import numpy as np
from scipy.spatial import Delaunay

from fsai import geometry
from fsai.objects.cone import Cone, CONE_COLOR_BIG_ORANGE, CONE_COLOR_BLUE, CONE_COLOR_YELLOW, CONE_COLOR_ORANGE


def create_boundary(
    blue_cones: List[Tuple[float, float]] = None,
    yellow_cones: List[Tuple[float, float]] = None,
    orange_cones: List[Tuple[float, float]] = None,
    big_cones: List[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if blue_cones is None: blue_cones = []
    if yellow_cones is None: yellow_cones =  []
    if orange_cones is None: orange_cones =  []
    if big_cones is None: big_cones =  []

    blue_cones = [Cone(pos, CONE_COLOR_BLUE) for pos in blue_cones]
    yellow_cones = [Cone(pos, CONE_COLOR_YELLOW) for pos in yellow_cones]
    orange_cones = [Cone(pos, CONE_COLOR_ORANGE) for pos in orange_cones]
    big_cones = [Cone(pos, CONE_COLOR_BIG_ORANGE) for pos in big_cones]

    big_cones = __merge_big_cones(big_cones)

    delaunay = __get_delaunay_triangles(
        blue_cones + yellow_cones + orange_cones + big_cones
    )

    blue_boundaries = []
    yellow_boundaries = []
    orange_boundaries = []

    track_graph: Dict[Cone: Set[Cone]] = {}

    for triangle in delaunay:
        for i in range(len(triangle)):
            if triangle[i] not in track_graph:
                track_graph[triangle[i]] = set()
            track_graph[triangle[i]].add(triangle[i - 1])
            track_graph[triangle[i]].add(triangle[i - 2])

    for cone in track_graph:
        if cone.color == CONE_COLOR_BLUE:
            for connection in track_graph[cone]:
                if connection.color == CONE_COLOR_BLUE:
                    blue_boundaries.append([cone.pos[0], cone.pos[1], connection.pos[0], connection.pos[1]])

        if cone.color == CONE_COLOR_YELLOW:
            for connection in track_graph[cone]:
                if connection.color == CONE_COLOR_YELLOW:
                    yellow_boundaries.append([cone.pos[0], cone.pos[1], connection.pos[0], connection.pos[1]])

        if cone.color == CONE_COLOR_BIG_ORANGE:
            cones = [{
                "cone": connection,
                "length": geometry.distance(connection.pos, cone.pos)
            } for connection in track_graph[cone]]
            cones = sorted(cones, key=lambda item: item['length'])
            closest_blue, closest_yellow = [], []
            for closest_cone in cones:
                if closest_cone["cone"].color == CONE_COLOR_YELLOW: closest_yellow.append(closest_cone["cone"])
                if closest_cone["cone"].color == CONE_COLOR_BLUE: closest_blue.append(closest_cone["cone"])

                if len(closest_blue) >= 2:
                    orange_boundaries.append([closest_blue[0].pos[0], closest_blue[0].pos[1], cone.pos[0], cone.pos[1]])
                    orange_boundaries.append([closest_blue[1].pos[0], closest_blue[1].pos[1], cone.pos[0], cone.pos[1]])
                    break
                if len(closest_yellow) >= 2:
                    orange_boundaries.append([closest_yellow[0].pos[0], closest_yellow[0].pos[1], cone.pos[0], cone.pos[1]])
                    orange_boundaries.append([closest_yellow[1].pos[0], closest_yellow[1].pos[1], cone.pos[0], cone.pos[1]])
                    break

    return blue_boundaries, yellow_boundaries, orange_boundaries


def get_delaunay_triangles(
    blue_cones: List[Tuple[float, float]] = None,
    yellow_cones: List[Tuple[float, float]] = None,
    orange_cones: List[Tuple[float, float]] = None,
    big_cones: List[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if blue_cones is None: blue_cones = []
    if yellow_cones is None: yellow_cones =  []
    if orange_cones is None: orange_cones =  []
    if big_cones is None: big_cones =  []

    blue_cones = [Cone(pos, CONE_COLOR_BLUE) for pos in blue_cones]
    yellow_cones = [Cone(pos, CONE_COLOR_YELLOW) for pos in yellow_cones]
    orange_cones = [Cone(pos, CONE_COLOR_ORANGE) for pos in orange_cones]
    big_cones = [Cone(pos, CONE_COLOR_BIG_ORANGE) for pos in big_cones]

    big_cones = __merge_big_cones(big_cones)

    delaunay = __get_delaunay_triangles(
        blue_cones + yellow_cones + orange_cones + big_cones
    )

    return [[x[0].pos, x[1].pos, x[2].pos] for x in delaunay]


def __get_delaunay_triangles(all_cones):
    triangles, invalid = [], []

    delaunay_triangles = __get_delaunay_triangulations(all_cones)
    missed_cones = set(all_cones)

    for triangle in delaunay_triangles:
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


def __get_delaunay_triangulations(all_cones: List[Cone]) -> List[List[Cone]]:
    """
    Create the delaunay triangles for the given cones
    :param all_cones: All the known cones in the map
    :return: List of lists of cones representing the list of triangles
    """

    delaunay = Delaunay(np.asarray([cone.pos for cone in all_cones]))

    # convert triangles from list of indexes to list of cones
    delaunay = [[all_cones[tri[0]], all_cones[tri[1]], all_cones[tri[2]]] for tri in delaunay.simplices]
    return delaunay


def __merge_big_cones(big_cones: List[Cone]):
    """
    In the FS-AI events, the starting big orange cones come in pairs, however these pairs are essentially treated as
    a single marker which denotes the start line. This method will combine the pairs of cones into discrete markers.

    :param big_cones: List of the big orange cones
    :return: Markers representing the start line markers
    """
    merged_orange_cones: List[Cone] = []
    for cone in big_cones:
        too_close = False
        for merged_cone in merged_orange_cones:
            too_close = too_close or geometry.distance(cone.pos, merged_cone.pos) < 2.75
        if not too_close:
            merged_orange_cones += [cone]
    return merged_orange_cones
