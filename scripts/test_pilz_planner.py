#!/usr/bin/env python3

import rospy
import sys
from geometry_msgs.msg import PoseStamped
from isaac_ros_bridge.srv import PlanPose, PlanPoseRequest

def test_pilz_planner():
    """Test different Pilz Industrial Motion Planner types"""
    
    rospy.init_node('test_pilz_planner')
    
    # Wait for the planning service
    rospy.wait_for_service('/plan_pose')
    plan_service = rospy.ServiceProxy('/plan_pose', PlanPose)
    
    # Test pose (adjust based on your robot's workspace)
    test_pose = PoseStamped()
    test_pose.header.frame_id = "world"
    test_pose.pose.position.x = 0.5
    test_pose.pose.position.y = 0.0
    test_pose.pose.position.z = 0.5
    test_pose.pose.orientation.w = 1.0
    
    # Test different Pilz planner types
    pilz_types = ["PTP", "LIN", "CIRC"]
    
    for planner_type in pilz_types:
        print(f"\n=== Testing Pilz {planner_type} Planner ===")
        
        # Set the planning pipeline and type
        rospy.set_param('/moveit_planning_pipeline', 'pilz_industrial_motion_planner')
        rospy.set_param('/moveit_planner_type', planner_type)
        
        # Create planning request
        req = PlanPoseRequest()
        req.group_name = "manipulator"  # Adjust to your robot's group name
        req.goal = test_pose
        
        try:
            # Call planning service
            response = plan_service(req)
            
            if response.success:
                print(f"✓ {planner_type} planning successful!")
                print(f"  Trajectory points: {len(response.trajectory.points)}")
                if len(response.trajectory.points) > 0:
                    print(f"  Duration: {response.trajectory.points[-1].time_from_start.to_sec():.2f}s")
            else:
                print(f"✗ {planner_type} planning failed: {response.error_message}")
                
        except rospy.ServiceException as e:
            print(f"✗ Service call failed: {e}")
        
        rospy.sleep(1.0)  # Wait between tests
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    try:
        test_pilz_planner()
    except rospy.ROSInterruptException:
        pass
