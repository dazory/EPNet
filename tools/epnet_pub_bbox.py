import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
from epnet_carla.msg import kittiMsg, kittiMsgs
from visualization_msgs.msg import Marker, MarkerArray
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox


class EPNetCarlaViz:

    def __init__(self):
        '''
        필요한 것
        input_data = { 'pts_input': inputs }
        input_data['pts_origin_xy']
        input_data['img'] = img

        '''
        rospy.init_node('visualize_epnet')

        # self.img_sub = rospy.Subscriber("/carla/ego_vehicle/camera/rgb/camera1/image_color", Image, self.callback_image)
        # self.lidar_sub = rospy.Subscriber("/carla/ego_vehicle/lidar/lidar1/point_cloud", PointCloud2, self.callback_lidar)
        self.pred_pub = rospy.Subscriber('/carla/prediction', kittiMsgs, self.callback_viz)  # String --> CUSTOM MSG TYPE
        self.pub = rospy.Publisher("bbox", BoundingBoxArray, queue_size=10)

        self.r = rospy.Rate(24)

    def callback_viz(self, kittis):
        print("callback!")
        box_arr = BoundingBoxArray()
        for kitti in kittis.kittis:
            box_a = BoundingBox()
            box_a.label = 1
            box_a.dimensions.x = 1
            box_a.dimensions.y = 1
            box_a.dimensions.z = 1
            box_a.header.stamp = rospy.Time.now()

            box_a.pose.position.x = kitti.location[0]
            box_a.pose.position.y = kitti.location[1]
            box_a.pose.position.z = kitti.location[2]
            box_arr.boxes.append(box_a)

            # marker.type = Marker.CUBE
            # marker.action = Marker.ADD

            # marker.pose.orientation.x = 0.0;
            # marker.pose.orientation.y = 0.0;
            # marker.pose.orientation.z = 0.0;
            # marker.pose.orientation.w = 1.0;
            # marker.scale.x = 1;
            # marker.scale.y = 0.1;
            # marker.scale.z = 0.1;
            # marker.color.a = 1.0;
            # marker.color.r = 0.0;
            # marker.color.g = 1.0;
            # marker.color.b = 0.0;
        self.pub.publish(box_arr)
        # self.r.sleep()


if __name__ == '__main__':
    epnetcarlaviz = EPNetCarlaViz()
    rospy.spin()