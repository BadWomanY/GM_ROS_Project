#!/usr/bin/python3
import sys
import rospy
import moveit_commander

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory
from std_srvs.srv import Trigger, TriggerResponse
from isaac_ros_bridge.srv import PlanPose, PlanPoseResponse

scene = None
_known_ids = set()

def _pose_from_dict(d):
    ps = PoseStamped()
    ps.header.frame_id = d.get("frame_id", "world")
    p = d["position"]; q = d["orientation"]
    ps.pose.position.x = p["x"]; ps.pose.position.y = p["y"]; ps.pose.position.z = p["z"]
    ps.pose.orientation.x = q["x"]; ps.pose.orientation.y = q["y"]; ps.pose.orientation.z = q["z"]; ps.pose.orientation.w = q["w"]
    return ps

def sync_param_boxes():
    """Idempotently sync boxes from /isaac/moveit_boxes into the planning scene."""
    global scene, _known_ids
    if scene is None:
        scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        rospy.sleep(0.5)

    boxes = rospy.get_param("/isaac/moveit_boxes", [])
    want_ids = set()

    for b in boxes:
        bid = b["id"]; want_ids.add(bid)
        size = b["size"]              # [sx, sy, sz]
        pose = _pose_from_dict(b["pose"])
        scene.add_box(bid, pose, size=size)  # add/update

    # remove stale
    for rid in list(_known_ids - want_ids):
        scene.remove_world_object(rid)

    _known_ids = want_ids
    return list(want_ids)

def _wait_scene(ids, timeout=2.0):
    """Wait until all ids are visible to the PlanningSceneInterface (RViz will show them too)."""
    start = rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec() - start < timeout:
        present = set(scene.get_known_object_names())
        if set(ids).issubset(present):
            return True, present
        rospy.sleep(0.05)
    return False, set(scene.get_known_object_names())

def handle_update_boxes(_req):
    try:
        ids = sync_param_boxes()
        ok, present = _wait_scene(ids, timeout=2.0)
        msg = "Synced boxes: {}. Present: {}".format(ids, list(present))
        rospy.loginfo(msg)
        return TriggerResponse(success=ok, message=msg)
    except Exception as e:
        return TriggerResponse(success=False, message=f"exception: {e}")

def handle_plan(req: PlanPose) -> PlanPoseResponse:
    try:
        # ensure scene is current even if client forgot to call /update_boxes
        sync_param_boxes()

        group = moveit_commander.MoveGroupCommander(req.group_name)
        group.set_planner_id("RRTstarkConfigDefault")  # Use RRT* planner
        group.set_pose_reference_frame("world")
        group.set_start_state_to_current_state()
        group.set_max_velocity_scaling_factor(0.07)      # 0.1% of max velocity (very slow)
        group.set_max_acceleration_scaling_factor(0.07)  # 0.1% of max acceleration (very slow)
        group.set_planning_time(3.0)
        group.set_num_planning_attempts(10)

        group.set_pose_target(req.goal)

        ok, plan, _, _ = group.plan()
        if not ok:
            return PlanPoseResponse(False, JointTrajectory(), "planning failed")

        return PlanPoseResponse(True, plan.joint_trajectory, "ok")
    except Exception as e:
        return PlanPoseResponse(False, JointTrajectory(), f"exception: {e}")

if __name__ == "__main__":
    rospy.init_node("moveit_planner_server")
    moveit_commander.roscpp_initialize(sys.argv)
    scene = moveit_commander.PlanningSceneInterface(synchronous=True)
    rospy.sleep(0.5)

    # NEW: service to force-sync & visualize boxes
    rospy.Service("/update_boxes", Trigger, handle_update_boxes)

    rospy.Service("/plan_pose", PlanPose, handle_plan)
    rospy.loginfo("moveit_planner_server ready")
    rospy.spin()
