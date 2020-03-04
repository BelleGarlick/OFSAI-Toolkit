from typing import List

from fsai.mapping.cluster import Cluster
from fsai.objects.cone import Cone
from fsai.objects.point import Point


class Map:
    def __init__(self):
        # store all the active clusters in this list
        self.clusters: List[Cluster] = []

        # if a cluster gets too big then it will have a lot of points and computation will grow. In order to minimize
        # this computation we can lock a cone. Additionally, if a cluster has say 50 points, the 51st point will not
        # affect the mean of the cluster too much. Once locked a cluster can exists just as a cone within this list.
        # if a new cone gets close enough to this cluster then we ignore it, and we don't keep re-calculating the
        # locked cones in order to keep computation low.
        self.locked_clusters: List[Cone] = []

        # turning variables
        self.MIN_DISTANCE = 1  # min distance a cone can be to a cluster to be in that cluster (m)
        self.MAX_CLUSTER_SIZE = 30  # max size a cluster can be before we lock it for being too large
        self.MAX_EPOCHS = 5  # maximum times the re-clustering can be called per re-cluster loop
        # preventing erogenous clusters being considered true cones we check that the
        # cluster has enough points to be considered a true cone
        self.MINIMUM_CLUSTER_SIZE = 5

    def map(self, car_position: Point, car_angle: float, local_cones: List[Cone]) -> None:
        """
        This function should be called every frame the car gets input.
        Given the car position, car angle, and the known cones in local space, this function will
        attempt to user k means clustering to store all the cones in order to stabilise everything.
        :param car_position: Euclidean co-ordinates of the car in x/y
        :param car_angle: Angle of the car (radians)
        :param local_cones: Positions of cones the car can see relative to the car. A cone directly ahead of the car
                            should be (0, 1) regardless of the car's position
        :return: None
        """

        # loop though all cones, convert them into local space then add them to the cones
        for cone in local_cones:
            cone.pos.rotate_around(Point(0, 0), car_angle)
            cone.pos.add(car_position)

            # first we need to check if the nearest locked cone is close enough to the current cone. If so
            # then we can ignore this cone and assume it's in the cluster that is now locked, and can therefore
            # just ignore the cone
            nearest_locked_cone = self.__get_nearest_locked_cluster(cone.pos)
            if nearest_locked_cone is not None and nearest_locked_cone.pos.distance(cone.pos) < self.MIN_DISTANCE:
                # cone is close enough to the closest locked cone that we can ignore it
                pass
            else:
                # if the cone is not close to a cluster then we can just add the cone to any cluster
                # then let the k-mean clustering alg deal with sorting out the clusters
                if len(self.clusters) == 0:
                    self.clusters.append(Cluster(cone))
                else:
                    self.clusters[0].points.append(cone)

        # run the re-clustering loops
        self.__recluster_loop()
        if self.__miosis():
            self.__recluster_loop()

        # lock clusters and remove empty clusters
        self.__clean_up_clusters()

    def __clean_up_clusters(self) -> None:
        # Remove / Lock clusters
        clusters_to_remove: List[Cluster] = []  # buffer to store all clusters to remove whilst iterating on the list
        for cluster in self.clusters:
            # if clusters get too large then we lock it
            # NOTE: you could also lock a cluster if it's too far from the car and large enough - up to you
            if len(cluster.points) > self.MAX_CLUSTER_SIZE:
                self.locked_clusters.append(Cone(pos=cluster.position, color=cluster.color))
                clusters_to_remove.append(cluster)

            # if cluster is empty delete it
            if len(cluster.points) == 0:
                clusters_to_remove.append(cluster)

        for cluster in clusters_to_remove:
            self.clusters.remove(cluster)

    def __recluster_loop(self) -> None:
        """
        This function should be called to start the k-means clustering alg.
        Every time the clustering call function is run, an error value will be returned which is the sum of distance
        each cluster has moved. This can be used to gauge how much the clusters are moving. If the number is really
        large then the clusters have moved a lot.
        Once the error is small enough or enough epochs have been called then the function will end
        :return: None
        """
        error: float = 1
        epoch: int = 0
        while error > 0.01 and epoch < self.MAX_EPOCHS:
            error = self.__recluster()
            epoch += 1

    def __miosis(self) -> bool:
        """
        This function will call to check if any clusters are large enough that they split apart into new clusters
        This happens by looking by searching for the cone that is furthest away from the mean of the cluster. Then
        a new cluster can be created for just that cone. After miosis is called (if a new cluster has been created),
        then the cones will be re-clustered in order to add more cones the this new cluster with the one cone
        :return: Bool: Has a new cluster been created
        """
        new_clusters: List[Cluster] = []  # list all the new clustered created
        for cluster in self.clusters:
            if len(cluster.points) != 0:
                # store info about the cluster
                avg_distance = 0
                furthest_point, furthest_distance = None, -1

                # check the metrics for each cone in the cluster
                for cone in cluster.points:
                    # add the distance if the cone's distance to the avg error
                    distance_to_cluster = cone.point.distance(cluster.position)
                    avg_distance += distance_to_cluster

                    # if the displacement is higher than the further value, then update the furthest value
                    if distance_to_cluster > furthest_distance:
                        furthest_distance = distance_to_cluster
                        furthest_point = cone

                # calc average
                avg_distance /= len(cluster.points)

                # if the average cone error is too large then move the cone to a different cluster
                if avg_distance > self.MIN_DISTANCE / 2:
                    new_clusters.append(Cluster(furthest_point.point))
                    cluster.points.remove(furthest_point)

        # add the new clusters to the current clusters
        self.clusters = self.clusters + new_clusters

        # return true if new clusters were created
        return len(new_clusters) != 0

    def __recluster(self) -> float:
        """
        This function should be called to re-cluster all clusters.
        This means all points will be moved to make sure all cones are in the clusters they're closest too.
        :return: Sum how how far each cluster's center has moved after being re-calculated
        """
        for cluster in self.clusters:  # loop through each cluster, then each point in each cluster
            moved_cones: List[Cone] = []  # stores a list of all moved cones to remove from the cluster

            for cone in cluster.points:
                # calculate the distance of each cone to it's custer's mean position
                cone_displacement = cone.pos.distance(cluster.position)
                closest_cluster: Cluster = cluster

                # loop thought each cluster to find if there is a cluster that is closer to the cone in order to move it
                for potential_cluster in self.clusters:
                    distance_to_new_cluster = cone.pos.distance(potential_cluster.position)
                    if distance_to_new_cluster < cone_displacement:  # if cone is closer to this cluster, move the cone
                        closest_cluster = potential_cluster
                        cone_displacement = distance_to_new_cluster

                # if cone is closer to a different cluster
                if closest_cluster != cluster:
                    moved_cones.append(cone)  # add to the new cluster
                    closest_cluster.points.append(cone)  # add to the points to remove

            # remove each buffered cone from the list
            for moved_cone in moved_cones:
                cluster.points.remove(moved_cone)

        # re-calculate cluster centers
        total_error = 0
        for cluster in self.clusters:
            total_error += cluster.recalculate()

        # return the sum of how far each cluster has moved (m)
        return total_error

    def __get_nearest_locked_cluster(self, p: Point) -> Cone:
        """
        Get the nearest locked cluster points to point p
        :param p: The point to get the closest cluster too
        :return: returns the locked clusters that is closest
        """
        nearest_locked_cluster = self.locked_clusters[0]
        cluster_distance = nearest_locked_cluster.pos.distance(p)

        # loop thought all clusters to find the closets point
        for cluster in self.locked_clusters:
            current_distance = p.distance(cluster.pos)

            # if cluster distance is larger then update the closest point
            if current_distance < cluster_distance:
                nearest_locked_cluster = cluster
                cluster_distance = current_distance

        # returns the nearest locked cluster
        return nearest_locked_cluster

    def get_clusters_global(self) -> List[Cone]:
        """
        Get all the clusters in the class in global space
        :return: List of all clusters (as cones)
        """
        data_points: List[Cone] = list(self.locked_clusters)

        if len(self.clusters) + len(self.locked_clusters) > 2:
            for cluster in self.clusters:
                if len(cluster.points) > self.MINIMUM_CLUSTER_SIZE:
                    data_points.append(Cone(pos=cluster.position, color=cluster.color))
        else:
            for cluster in self.clusters:
                data_points.append(Cone(pos=cluster.position, color=cluster.color))
        return data_points

    def get_clusters_relative(self, car_pos: Point, car_orientation: float) -> List[Cone]:
        """
        Get a list of all the cones relative to the car
        :param car_pos: Position of the car
        :param car_orientation: Orientation of the car (radians)
        :return: List of all cones relative to the car
        """
        relative_points: List[Cone] = []
        for cone in self.get_clusters_global():
            # deep copy the object so that as values are mutated the original copies aren't affected
            cone = cone.copy()

            # rotate the point around the car
            cone.point.sub(car_pos)
            cone.point.rotate_around(car_pos, -car_orientation)

            relative_points.append(cone)
        return relative_points
