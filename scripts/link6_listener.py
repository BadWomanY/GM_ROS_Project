#!/usr/bin/env python
import rospy
import tf
from geometry_msgs.msg import PoseStamped

def main():
    rospy.init_node('link6_pose_listener', anonymous=True)
    listener = tf.TransformListener()
    pub = rospy.Publisher('/link6_pose', PoseStamped, queue_size=10)
    rate = rospy.Rate(30)  # 30 Hz

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('base_link', 'link6', rospy.Time(0))

            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = 'base_link'

            pose_msg.pose.position.x = trans[0]
            pose_msg.pose.position.y = trans[1]
            pose_msg.pose.position.z = trans[2]

            pose_msg.pose.orientation.x = rot[0]
            pose_msg.pose.orientation.y = rot[1]
            pose_msg.pose.orientation.z = rot[2]
            pose_msg.pose.orientation.w = rot[3]

            pub.publish(pose_msg)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        rate.sleep()

if __name__ == '__main__':
    main()
