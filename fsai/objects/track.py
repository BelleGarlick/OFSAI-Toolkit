import json
from typing import List

from fsai.mapping.boundary_estimation import create_boundary
from fsai.objects.cone import Cone, CONE_COLOR_BLUE, CONE_COLOR_YELLOW, CONE_COLOR_ORANGE, CONE_COLOR_BIG_ORANGE


class Track:
    def __init__(self, path: str = None):
        self.blue_cones: List[Cone] = []
        self.yellow_cones: List[Cone] = []
        self.orange_cones: List[Cone] = []
        self.big_orange_cones: List[Cone] = []

        # Load the track from json
        if path is not None:
            self.load_track(path)

    def load_track(self, path: str):
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
            if "big_orange_cones" in track_json:
                self.big_orange_cones = [
                    Cone(x=c["x"], y=c["y"], color=CONE_COLOR_BIG_ORANGE) for c in track_json["big_orange_cones"]]

    def to_json(self):
        return {
            "blue_cones": [{"x": c.pos.x, "y": c.pos.y} for c in self.blue_cones],
            "yellow_cones": [{"x": c.pos.x, "y": c.pos.y} for c in self.yellow_cones],
            "orange_cones": [{"x": c.pos.x, "y": c.pos.y} for c in self.orange_cones],
            "big_orange_cones": [{"x": c.pos.x, "y": c.pos.y} for c in self.big_orange_cones]
        }

    def save_track(self, output_path: str):
        with open(output_path, "w+") as file:
            file.write(json.dumps(self.to_json(), indent=4))

    def get_boundary(self):
        return create_boundary(
            blue_cones=self.blue_cones,
            yellow_cones=self.yellow_cones,
            orange_cones=self.orange_cones,
            big_orange_cones=self.big_orange_cones
        )

