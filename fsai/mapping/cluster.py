from typing import List

from fsai.objects.cone import Cone, CONE_COLOR_BLUE, CONE_COLOR_YELLOW, CONE_COLOR_ORANGE, CONE_COLOR_BIG_ORANGE
from fsai.objects.point import Point


class Cluster:
    def __init__(self, initial_point: Cone):
        self.position: Point = initial_point.point
        self.points: List[Cone] = [initial_point]
        self.color: int = CONE_COLOR_BLUE

    def recalculate(self):
        """
        Recalculate the colour and mean of this cluster
        :return: how much the mean has moved
        """
        error = 0
        if len(self.points) != 0:
            average_point = Point(0, 0)

            # loop through each point in the current cluster
            blue_count, yellow_count, orange_count, big_orange_count = 0, 0, 0, 0
            for cone in self.points:
                average_point.add(cone.point)

                # increment the count of each colour depending on the colour of the cone
                blue_count += cone.color == CONE_COLOR_BLUE
                yellow_count += cone.color == CONE_COLOR_YELLOW
                orange_count += cone.color == CONE_COLOR_ORANGE
                big_orange_count += cone.color == CONE_COLOR_BIG_ORANGE

            # get the average off
            average_point.x /= len(self.points)
            average_point.y /= len(self.points)

            # update the error and position
            error = self.position.distance(average_point)
            self.position = average_point

            # set the colour of this cluster based upon the amount of colours in this cluster
            if CONE_COLOR_BIG_ORANGE > yellow_count and CONE_COLOR_BIG_ORANGE > blue_count:
                self.color = CONE_COLOR_BIG_ORANGE
            elif CONE_COLOR_ORANGE > yellow_count and CONE_COLOR_ORANGE > blue_count:
                self.color = CONE_COLOR_ORANGE
            elif yellow_count > blue_count:
                self.color = CONE_COLOR_YELLOW
            else:
                self.color = CONE_COLOR_BLUE

        return error
