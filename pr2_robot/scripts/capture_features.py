#!/usr/bin/env python
import numpy as np
import pickle
import rospy

from pr2_robot.pcl_helper import *
from pr2_robot.training_helper import spawn_model
from pr2_robot.training_helper import delete_model
from pr2_robot.training_helper import initial_setup
from pr2_robot.training_helper import capture_sample
from pr2_robot.features import compute_color_histograms
from pr2_robot.features import compute_normal_histograms
from pr2_robot.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    rospy.init_node('capture_node')

    pcl_sample_cloud_pub = rospy.Publisher("/pcl_sample_cloud", PointCloud2, queue_size=1)

    # TODO(saminda): get this parameter from a parameter server and automate
    training_set_name = 'list3'
    num_samples = 256

    models_dict = {
        'list1': ['biscuits', 'soap', 'soap2'],
        'list2': ['book', 'glue'],
        'list3': ['sticky_notes', 'snacks', 'eraser'],
        'list4': ['biscuits', 'soap', 'soap2', 'book', 'glue', 'sticky_notes', 'snacks', 'eraser']
    }

    # ['biscuits', 'soap', 'soap2', 'book', 'glue']
    # ['biscuits', 'soap', 'soap2', 'book', 'glue', 'sticky_notes', 'snacks', 'eraser']


    rospy.loginfo("capture_node_activated: {}".format(training_set_name))

    # object_list_param = rospy.get_param('/object_list')

    # for i in range(len(object_list_param)):
    #     object_name = object_list_param[i]['name']
    #     object_group = object_list_param[i]['group']
    #     rospy.loginfo("name: {} group: {}".format(object_name, object_group))

    #
    # models = [ \
    #     'beer',
    #     'bowl',
    #     'create',
    #     'disk_part',
    #     'hammer',
    #     'plastic_cup',
    #     'soda_can']

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models_dict[training_set_name]:
        rospy.loginfo("model_name: {}".format(model_name))

        spawn_model(model_name)

        rospy.loginfo("num_samples: {}".format(num_samples))
        for i in range(num_samples):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()

                # same pipeline as processing to make sure that training dist
                # is similar to test dist
                cloud = ros_to_pcl(sample_cloud)

                # Statistical outlier filter
                fil = cloud.make_statistical_outlier_filter()
                fil.set_mean_k(15)
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
                sample_cloud = vox.filter()

                sample_cloud_arr = sample_cloud.to_array()
                ##

                #sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # pub
                # pcl_sample_cloud_pub.publish(sample_cloud)
                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True
                    # Extract histogram features
                    # Convert back to ros for helper functions
                    sample_cloud = pcl_to_ros(sample_cloud)

                    chists = compute_color_histograms(sample_cloud, using_hsv=True)
                    normals = get_normals(sample_cloud)
                    nhists = compute_normal_histograms(normals)
                    feature = np.concatenate((chists, nhists))
                    labeled_features.append([feature, model_name])
                    # debugging only
                    # pcl_sample_cloud_pub.publish(sample_cloud)

            if i % 5 == 0:
                print("model_name: {} i: {}/{}".format(model_name, i, num_samples))

        delete_model()

    rospy.loginfo("labeled_features: {}".format(len(labeled_features)))
    pickle.dump(labeled_features, open('training_set_{}_vox.sav'.format(training_set_name), 'wb'))
