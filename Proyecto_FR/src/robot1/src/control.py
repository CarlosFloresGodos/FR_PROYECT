#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from markers import *
from functions_dh import *
import numpy as np
from copy import copy

if __name__ == '__main__':

    # Initialize the node
    rospy.init_node("testKinematicControlPosition")
    print('starting motion ... ')
    # Publisher: publish to the joint_states topic
    pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    # Files for the logs
    fxcurrent = open("/home/user/xcurrent.txt", "w")                
    fxdesired = open("/home/user/xdesired.txt", "w")
    fq = open("/home/user/q.txt", "w")

    # Markers for the current and desired positions
    bmarker_current = BallMarker(color['RED'])
    bmarker_desired = BallMarker(color['GREEN'])

    # Joint names
    jnames = ['Slider_1', 'Revolute_2', 'Revolute_3', 'Slider_4', 'Revolute_6', 'Revolute_1']

    # Desired position
    xd = np.array([0.9, -1.2, 1.6])
    # Initial configuration
    q0 = np.array([0.0, 0.0, 0.0, 0, 0.0, 0.0])

    # Resulting initial position (end effector with respect to the base link)
    T = fkine(q0)
    x0 = T[0:3,3]

    # Red marker shows the achieved position
    bmarker_current.xyz(x0)
    # Green marker shows the desired position
    bmarker_desired.xyz(xd)

    # Instance of the JointState message
    jstate = JointState()
    # Values of the message
    jstate.header.stamp = rospy.Time.now()
    jstate.name = jnames
    # Add the head joint value (with value 0) to the joints
    jstate.position = q0

    # Frequency (in Hz) and control period 
    freq = 200
    dt = 1.0/freq
    rate = rospy.Rate(freq)

    # Initial joint configuration
    q = copy(q0)
    x = x0
    k = 0.9
    epsilon = 10e-4
    t = 0
    cnt = 0

    # Joint limits for q[0] and q[3]
    q0_limits = [0.0, 0.50]  # Example: Slider_1 limits
    q3_limits = [0.0, 0.35]  # Example: Slider_4 limits

    # Main loop
    while not rospy.is_shutdown():
        # Current time (needed for ROS)
        jstate.header.stamp = rospy.Time.now()
        
        # Kinematic control law for position
        J = jacobian(q)
        x = fkine(q)[0:3,3]
        e = x - xd
        
        if (np.linalg.norm(e) < epsilon):
            print("Se llegó al punto deseado en {:.3} segundos".format(cnt * dt))
            break

        de = -k * e
        dq = np.linalg.pinv(J).dot(de)
        q = q + dt * dq

        # Apply joint limits
        q[0] = np.clip(q[0], q0_limits[0], q0_limits[1])
        q[3] = np.clip(q[3], q3_limits[0], q3_limits[1])

        # Solamente para evitar un bucle infinito si algo sale mal
        cnt = cnt + 1
        t += 1
        if (cnt > 1e5):
            print("Se excedió el número de iteraciones")
            break
        
        # Log values                                                      
        fxcurrent.write(str(t) + ' ' + str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]) + '\n')
        fxdesired.write(str(t) + ' ' + str(xd[0]) + ' ' + str(xd[1]) + ' ' + str(xd[2]) + '\n')
        fq.write(str(t) + ' ' + str(q[0]) + " " + str(q[1]) + " " + str(q[2]) + " " + str(q[3]) + " " +
                 str(q[4]) + " " + str(q[5]) + "\n")
        
        # Publish the message
        jstate.position = q
        pub.publish(jstate)
        bmarker_desired.xyz(xd)
        bmarker_current.xyz(x)
        # Wait for the next iteration
        rate.sleep()

    print('ending motion ...')
    fxcurrent.close()
    fxdesired.close()
    fq.close()
    show_graph(1)
