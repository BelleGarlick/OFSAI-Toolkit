import json
from typing import List, Tuple

from fsai.mapping.boundary_estimation import create_boundary
from fsai.car.car import Car
from fsai.objects.cone import Cone, CONE_COLOR_BLUE, CONE_COLOR_YELLOW, CONE_COLOR_ORANGE, CONE_COLOR_BIG_ORANGE
from fsai.objects.line import Line
from fsai.objects.point import Point


class Track:
    def __init__(self, path: str = None):
        """
        This object can be constructed with a file path to call the load_track path upon.
        :param path: Path to load a track from.
        """
        self.blue_cones: List[Cone] = []
        self.yellow_cones: List[Cone] = []
        self.orange_cones: List[Cone] = []
        self.big_cones: List[Cone] = []

        self.cars: List[Car] = []

        # Load the track from json
        if path is not None:
            self.load_track(path)

    def load_track(self, path: str):
        """
        Load the track from the json file outlines in the spec
        :param path: File path to load from
        :return: None
        """
        with open(path) as file:
            track_json = json.loads(file.read())
            if "blue_cones" in track_json:
                self.blue_cones = [
                    Cone(x=c["x"], y=c["y"], color=CONE_COLOR_BLUE) for c in track_json["blue_cones"]]
            if "yellow_cones" in track_json:
                self.yellow_cones = [
                    Cone(x=c["x"], y=c["y"], color=CONE_COLOR_YELLOW) for c in track_json["yellow_cones"]]
            if "orange_cones" in track_json:
                self.orange_cones = [
                    Cone(x=c["x"], y=c["y"], color=CONE_COLOR_ORANGE) for c in track_json["orange_cones"]]
            if "big_cones" in track_json:
                self.big_cones = [
                    Cone(x=c["x"], y=c["y"], color=CONE_COLOR_BIG_ORANGE) for c in track_json["big_cones"]]

            if "cars" in track_json:
                for car_json in track_json["cars"]:
                    car = Car()
                    car.pos = Point(car_json["pos"]["x"], car_json["pos"]["y"])
                    car.orientation = car_json["orientation"]
                    self.cars.append(car)

    def to_json(self):
        """
        Convert object into the json/dict outlined in the README.md.
        :return: Json/dict representing this object.
        """
        return {
            "blue_cones": [{"x": c.pos.x, "y": c.pos.y} for c in self.blue_cones],
            "yellow_cones": [{"x": c.pos.x, "y": c.pos.y} for c in self.yellow_cones],
            "orange_cones": [{"x": c.pos.x, "y": c.pos.y} for c in self.orange_cones],
            "big_cones": [{"x": c.pos.x, "y": c.pos.y} for c in self.big_cones],
            "cars": [
                {"pos": {"x": car.pos.x, "y": car.pos.y}, "orientation": car.heading} for car in self.cars
            ]
        }

    def save_track(self, output_path: str):
        """
        Writes the json to the file path given.
        :param output_path: Output path to save the object too.
        :return: None
        """
        with open(output_path, "w+") as file:
            file.write(json.dumps(self.to_json(), indent=4))

    def get_boundary(self) -> Tuple[List[Line], List[Line], List[Line]]:
        """
        Get the boundary of the track using the fsai.mapping.boundary_estimation.create_boundary method.
        :return: Three lists of lines representing blue, yellow and orange boundaries respectively.
        """
        return create_boundary(
            blue_cones=self.blue_cones,
            yellow_cones=self.yellow_cones,
            orange_cones=self.orange_cones,
            big_cones=self.big_cones
        )

