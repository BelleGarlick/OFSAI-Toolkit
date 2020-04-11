import numpy as np


# Integer enums representing the colours of the cone
CONE_COLOR_BLUE = 0
CONE_COLOR_YELLOW = 1
CONE_COLOR_BIG_ORANGE = 2
CONE_COLOR_ORANGE = 3


class Cone:
    def __init__(
            self,
            pos: np.ndarray = np.zeros((0, 2)),
            color: int = CONE_COLOR_BLUE
    ):
        """
        Construct the cone object.

        cone = Cone() - Blue cone as 0, 0
        cone = Cone(x=9, y=5, color=CONE_COLOR_YELLOW) - Yellow cone at 9, 5
        cone = Cone(point=Point(1, 2), color=CONE_COLOR_ORANGE) - Orange cone at 1, 2

        :param x: Optional x position of the cone
        :param y: Optional y position of the cone
        :param pos: Optional position of the cone
        :param color: Color of the cone, represented as an int
        """
        self.pos = np.ndarray
        if pos is not None:
            self.pos = pos

        self.color = color

    def copy(self):
        """
        Create a new deep copy of the cone object
        :return: New mutable cone object
        """
        return Cone(self.pos, color=self.color)

    def __str__(self):
        return "Point: ({}), Color: {}".format(self.pos, self.color)
