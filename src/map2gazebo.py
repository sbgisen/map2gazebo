#!/usr/bin/env python

import os

import cv2
import numpy as np
import rospy
import trimesh
from nav_msgs.msg import OccupancyGrid


class MapConverter(object):
    def __init__(self, map_topic, threshold=1, height=2.0, use_contours=False):
        self.test_map_pub = rospy.Publisher(
            "test_map", OccupancyGrid, latch=True, queue_size=1)
        self.threshold = threshold
        self.height = height
        self.use_contours = use_contours
        # Probably there's some way to get trimesh logs to point to ROS
        # logs, but I don't know it.  Uncomment the below if something
        # goes wrong with trimesh to get the logs to print to stdout.
        # trimesh.util.attach_to_log()
        map_msg = rospy.wait_for_message(map_topic, OccupancyGrid)
        self.create_map(map_msg)

    def create_map(self, map_msg):
        # Check that the saving directory exists
        mesh_type = rospy.get_param("~mesh_type", "stl")
        export_dir = rospy.get_param("~export_dir")
        file_name = rospy.get_param("~file_name", "map")
        if mesh_type not in ["stl", "dae"]:
            rospy.logerr("Invalid mesh type: {}".format(mesh_type))
            return
        if not os.path.exists(export_dir):
            rospy.logerr(f"Export directory {export_dir} does not exist")
            return
        rospy.loginfo("Received map")
        map_dims = (map_msg.info.height, map_msg.info.width)
        map_array = np.array(map_msg.data).reshape(map_dims)

        # set all -1 (unknown) values to 0 (unoccupied)
        map_array[map_array < 0] = 0
        if self.use_contours:
            contours = self.get_occupied_regions(map_array)
            meshes = [self.contour_to_mesh(c, map_msg.info) for c in contours]
            corners = list(np.vstack(contours))
            corners = [c[0] for c in corners]
            self.publish_test_map(corners, map_msg.info, map_msg.header)
        else:
            meshes = self.image_pixels_to_mesh(map_array, map_msg.info)

        mesh = trimesh.util.concatenate(meshes)

        # Export as STL or DAE
        if mesh_type == "stl":
            with open(export_dir + "/" + file_name + ".stl", 'wb') as f:
                mesh.export(f, "stl")
            rospy.loginfo("Exported STL: " + export_dir + "/" + file_name + ".stl")
        elif mesh_type == "dae":
            with open(export_dir + "/" + file_name + ".dae", 'w') as f:
                f.write(trimesh.exchange.dae.export_collada(mesh))
            rospy.loginfo("Exported DAE.")

    def publish_test_map(self, points, metadata, map_header):
        """
        For testing purposes, publishes a map highlighting certain points.
        points is a list of tuples (x, y) in the map's coordinate system.
        """
        test_map = np.zeros((metadata.height, metadata.width))
        for x, y in points:
            test_map[y, x] = 100
        test_map_msg = OccupancyGrid()
        test_map_msg.header = map_header
        test_map_msg.header.stamp = rospy.Time.now()
        test_map_msg.info = metadata
        test_map_msg.data = list(np.ravel(test_map))
        self.test_map_pub.publish(test_map_msg)

    def get_occupied_regions(self, map_array):
        """
        Get occupied regions of map
        """
        map_array = map_array.astype(np.uint8)
        _, thresh_map = cv2.threshold(
            map_array, self.threshold, 100, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            thresh_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # Using cv2.RETR_CCOMP classifies external contours at top level of
        # hierarchy and interior contours at second level.
        # If the whole space is enclosed by walls RETR_EXTERNAL will exclude
        # all interior obstacles e.g. furniture.
        # https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        hierarchy = hierarchy[0]
        corner_idxs = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
        return [contours[i] for i in corner_idxs]

    def contour_to_mesh(self, contour, metadata):
        height = np.array([0, 0, self.height])
        meshes = []
        for point in contour:
            x, y = point[0]
            vertices = []
            new_vertices = [
                coords_to_loc((x, y), metadata),
                coords_to_loc((x, y + 1), metadata),
                coords_to_loc((x + 1, y), metadata),
                coords_to_loc((x + 1, y + 1), metadata)]
            vertices.extend(new_vertices)
            vertices.extend([v + height for v in new_vertices])
            faces = [[0, 2, 4],
                     [4, 2, 6],
                     [1, 2, 0],
                     [3, 2, 1],
                     [5, 0, 4],
                     [1, 0, 5],
                     [3, 7, 2],
                     [7, 6, 2],
                     [7, 4, 6],
                     [5, 4, 7],
                     [1, 5, 3],
                     [7, 3, 5]]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not mesh.is_volume:
                rospy.logdebug("Fixing mesh normals")
                mesh.fix_normals()
            meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
        mesh.remove_duplicate_faces()
        # mesh will still have internal faces.  Would be better to get
        # all duplicate faces and remove both of them, since duplicate faces
        # are guaranteed to be internal faces
        return mesh

    def image_pixels_to_mesh(self, image, metadata):
        height = np.array([0, 0, self.height])
        vertices_dict = {}
        faces_set = set()

        def add_face(v0, v1, v2, v3, height_offset, reverse=False):
            v0 = tuple(v0 + height_offset)
            v1 = tuple(v1 + height_offset)
            v2 = tuple(v2 + height_offset)
            v3 = tuple(v3 + height_offset)
            if reverse:
                faces_set.add((v3, v1, v2))
                faces_set.add((v2, v1, v0))
            else:
                faces_set.add((v0, v1, v2))
                faces_set.add((v2, v1, v3))

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if image[y, x] < self.threshold:
                    continue

                v0 = coords_to_loc((x, y), metadata)
                v1 = coords_to_loc((x, y + 1), metadata)
                v2 = coords_to_loc((x + 1, y), metadata)
                v3 = coords_to_loc((x + 1, y + 1), metadata)

                if y == 0 or image[y - 1, x] < self.threshold:  # x+ neighbor
                    add_face(v0, v2, v0 + height, v2 + height, np.array([0, 0, 0]))
                if x == image.shape[1] - 1 or image[y, x + 1] < self.threshold:  # y+ neighbor
                    add_face(v2, v3, v2 + height, v3 + height, np.array([0, 0, 0]))
                if y == image.shape[0] - 1 or image[y + 1, x] < self.threshold:  # x- neighbor
                    add_face(v1, v3, v1 + height, v3 + height, np.array([0, 0, 0]), reverse=True)
                if x == 0 or image[y, x - 1] < self.threshold:  # y- neighbor
                    add_face(v0, v1, v0 + height, v1 + height, np.array([0, 0, 0]), reverse=True)

                # Roof face
                faces_set.add((tuple(v0 + height), tuple(v2 + height), tuple(v1 + height)))
                faces_set.add((tuple(v2 + height), tuple(v3 + height), tuple(v1 + height)))

                # Floor face
                faces_set.add((tuple(v0), tuple(v1), tuple(v2)))
                faces_set.add((tuple(v2), tuple(v1), tuple(v3)))

        vertices = set()
        faces = []
        for face in faces_set:
            face_indices = []
            for vertex in face:
                if vertex not in vertices_dict:
                    vertices_dict[vertex] = len(vertices_dict)
                face_indices.append(vertices_dict[vertex])
            faces.append(face_indices)

        vertices = np.array([list(vertex) for vertex in vertices_dict])
        faces = np.array(faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.remove_duplicate_faces()

        return mesh


def coords_to_loc(coords, metadata):
    x, y = coords
    loc_x = x * metadata.resolution + metadata.origin.position.x
    loc_y = y * metadata.resolution + metadata.origin.position.y
    # TODO: transform (x*res, y*res, 0.0) by Pose map_metadata.origin
    # instead of assuming origin is at z=0 with no rotation wrt map frame
    return np.array([loc_x, loc_y, 0.0])


def main():
    rospy.init_node("map2gazebo")
    map_topic = rospy.get_param("~map_topic", "map")
    occupied_thresh = rospy.get_param("~occupied_thresh", 1)
    box_height = rospy.get_param("~box_height", 2.0)
    use_contours = rospy.get_param("~use_contours", False)
    rospy.loginfo("map2gazebo running")
    MapConverter(map_topic, threshold=occupied_thresh, height=box_height, use_contours=use_contours)


if __name__ == "__main__":
    main()
