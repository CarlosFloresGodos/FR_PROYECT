#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState

from markers import *
from functions_dh import *
#from pyquaternion import Quaternion

if __name__ == '__main__':
 
 rospy.init_node("testForwardKine")
 pub = rospy.Publisher('joint_states', JointState, queue_size=10)
 bmarker = BallMarker(color['GREEN'])
 marker = FrameMarker()

 # Joint names
 jnames = ['Slider_1', 'Revolute_2', 'Revolute_3', 'Slider_4', 'Revolute_6', 'Revolute_1']
 # Joint Configuration
 q = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
 # End effector with respect to the base
 T = fkine(q)
 print(f"q0:{q[5]}  q1:{q[0]}   q2:{q[1]}   q3:{q[2]}   q4:{q[3]}   q5:{q[4]}")
 print("Valor obtenido\n", np.round(T, 3) )
 bmarker.position(T)
 
 x0 = TF2xyzquat(T)
 marker.setPose(x0)
 # Object (message) whose type is JointState
 jstate = JointState()
 # Set values to the message
 jstate.header.stamp = rospy.Time.now()
 jstate.name = jnames
 # Add the head joint value (with value 0) to the joints
 jstate.position = q
 
 # Loop rate (in Hz)
 rate = rospy.Rate(100)
 # Continuous execution loop
 while not rospy.is_shutdown():
  # Current time (needed for ROS)
  jstate.header.stamp = rospy.Time.now()
  # Publish the message
  pub.publish(jstate)
  bmarker.publish()
  marker.publish()
  # Wait for the next iteration
  rate.sleep()
  