#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from pr2_robot.features import compute_color_histograms
from pr2_robot.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from pr2_robot.marker_tools import *
from pr2_robot.msg import DetectedObjectsArray
from pr2_robot.msg import DetectedObject
from pr2_robot.pcl_helper import *

import rospy
import rospkg
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Generate output
def generate_output(centroids, indexer):
    dropbox_indexer = dict()
    for index, dropbox in enumerate(dropbox_param):
        dropbox_indexer[dropbox['group']] = index

    for group, index in dropbox_indexer.items():
        rospy.loginfo('group: {} index: {}'.format(group, index))
        rospy.loginfo('name: {} position: {}'.format(
            dropbox_param[index]['name'],
            dropbox_param[index]['position']))

    dict_list = []
    yaml_filename = None
    for object in object_list_param:
        object_name_value = object['name']
        object_group_value = object['group']
        rospy.loginfo("object_name_value: {} object_group_value: {}".format(object_name_value, object_group_value))

        if not object_name_value in indexer:
            rospy.logwarn("object_name_value: {} is not available in the indexer.".format(object_name_value))
            continue

        rospy.loginfo(
            "to yaml => object_name_value: {} object_group_value: {}".format(object_name_value, object_group_value))
        test_scene_num = Int32()
        # test case inference based on the objects detected
        if len(indexer) == 3:
            test_scene_num.data = 1
            yaml_filename = rospkg.RosPack().get_path('pr2_robot') + '/config/' + 'output_1.yaml'
        elif len(indexer) == 5:
            test_scene_num.data = 2
            yaml_filename = rospkg.RosPack().get_path('pr2_robot') + '/config/' + 'output_2.yaml'
        else:
            test_scene_num.data = 3
            yaml_filename = rospkg.RosPack().get_path('pr2_robot') + '/config/' + 'output_3.yaml'

        object_name = String()
        object_name.data = object_name_value
        rospy.loginfo('object_name: {}'.format(object_name.data))

        arm_name = String()
        arm_name.data = dropbox_param[dropbox_indexer[object_group_value]]['name']
        rospy.loginfo('arm_name: {}'.format(arm_name.data))

        pick_pose = Pose()
        pick_position = centroids[indexer[object_name_value]]
        rospy.loginfo('pick_position: {}'.format(pick_position))
        pick_pose.position.x = np.asscalar(pick_position[0])
        pick_pose.position.y = np.asscalar(pick_position[1])
        pick_pose.position.z = np.asscalar(pick_position[2])

        place_pose = Pose()
        place_position = dropbox_param[dropbox_indexer[object_group_value]]['position']
        rospy.loginfo('place_position: {}'.format(place_position))
        place_pose.position.x = float(place_position[0])
        place_pose.position.y = float(place_position[1])
        place_pose.position.z = float(place_position[2])

        # Populate various ROS messages
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

    rospy.loginfo("yaml_filename: {}".format(yaml_filename))
    send_to_yaml(yaml_filename, dict_list)


# Testing
# def pcl_test_callback(pcl_msg):
#     pcl_world_points_test_pub.publish(pcl_msg)
#     rospy.loginfo("pcl_test_callback")


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Statistical outlier filter
    fil = cloud.make_statistical_outlier_filter()
    fil.set_mean_k(10)
    fil.set_std_dev_mul_thresh(0.1)
    cloud_outlier_filter = fil.filter()

    # Voxel Grid Downsampling
    # Voxel Grid filter
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud_outlier_filter.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size
    # Experiment and find the appropriate size!

    # Set the voxel (or leaf) size
    vox.set_leaf_size(0.01, 0.01, 0.01)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    # filename = 'voxel_downsampled.pcd'
    # pcl.save(cloud_filtered, filename)

    # PassThrough Filter: z and y
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name('z')
    passthrough.set_filter_limits(0.6, 1.1)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()
    # filename = 'pass_through_filtered.pcd'
    # pcl.save(cloud_filtered, filename)
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name('y')
    passthrough.set_filter_limits(-0.5, 0.5)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    # RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    seg.set_distance_threshold(0.01)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    # Extract inliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    # filename = 'extracted_inliers.pcd'
    # pcl.save(extracted_inliers, filename)

    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    # filename = 'extracted_outliers.pcd'
    # pcl.save(extracted_outliers, filename)

    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cloud_objects = pcl_to_ros(extracted_outliers)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(1200)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    rospy.loginfo("cluster_color: {}".format(len(cluster_color)))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    rospy.loginfo("pcl_objects and pcl_table published")

    # Exercise-3:

    # Classify the clusters! (loop through each detected cluster one at a time)

    # Grab the points for the cluster

    # Compute the associated feature vector

    # Make the prediction

    # Publish a label into RViz

    # Add the detected object to the list of detected objects.
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_outliers.extract(pts_list)
        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # Complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        # points_arr = np.asarray([white_cloud[i] for i in pts_list])
        # rospy.loginfo("points_arr: {}".format(points_arr.shape))
        # label_pos = list(np.mean(points_arr, axis=0)[:3])
        # label_pos = list(white_cloud[pts_list[0]])
        # label_pos[2] += .3

        # object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    labels = []
    centroids = []  # to be list of tuples (x, y, z)
    indexer = dict()
    for index, object in enumerate(detected_objects):
        indexer[object.label] = index
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        points_centroid = np.mean(points_arr, axis=0)[:3]
        # Publish a label into RViz
        label_pos = list(points_centroid.copy())
        label_pos[2] += .3
        centroids.append(points_centroid)
        object_markers_pub.publish(make_label(object.label, label_pos, index))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Create the output
    generate_output(centroids, indexer)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        ##pr2_mover(detected_objects)
        rospy.loginfo("TODO: mover")
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

    # TODO: Get the PointCloud for a given object and obtain it's centroid

    # TODO: Create 'place_pose' for the object

    # TODO: Assign the arm to be used for pick_place

    # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

    # Wait for 'pick_place_routine' service to come up
    rospy.wait_for_service('pick_place_routine')

    try:
        pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        # TODO: Insert your message variables to be sent as a service request
        ##resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

        ##print ("Response: ",resp.success)

    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


# TODO: Output your request parameters into output yaml file


if __name__ == '__main__':

    # Initialize color_list
    get_color_list.color_list = []

    # ROS node initialization
    rospy.init_node('project_pr2_robot', anonymous=True)

    # get parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # Create Subscribers
    # Testing code
    # pcl_test_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_test_callback, queue_size=1)
    # pcl_world_points_test_pub = rospy.Publisher("/pcl_world_points_test", PointCloud2, queue_size=1)

    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # Here you need to create two publishers
    # Call them object_markers_pub and detected_objects_pub
    # Have them publish to "/object_markers" and "/detected_objects" with
    # Message Types "Marker" and "DetectedObjectsArray" , respectively
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model_path = rospkg.RosPack().get_path('pr2_robot') + '/scripts/' + 'model_list5_vox.sav'
    rospy.loginfo("model_path: {}".format(model_path))
    model = pickle.load(open(model_path, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    rospy.loginfo("model is loaded")

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
