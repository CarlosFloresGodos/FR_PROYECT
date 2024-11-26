#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions_dh import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    rospy.init_node("testInvKine")
    pub = rospy.Publisher('joint_states', JointState, queue_size=1)

    bmarker      = BallMarker(color['RED'])
    bmarker_des  = BallMarker(color['GREEN'])

    # Joint names
    jnames = ['Slider_1', 'Revolute_2', 'Revolute_3', 'Slider_4', 'Revolute_6', 'Revolute_1']
    # Desired position
    xd = np.array([0.5,-1,0.3])
    print(f"Posicion deseada: x: {xd[0]},y: {xd[1]},z: {xd[2]}")
    # Initial configuration
    q0 = np.array([0, 0.3, pi, 0.05,pi,0.1])
    # Inverse kinematic
    q,e= ikine(xd, q0)

    # Resulting position (end effector with respect to the base link)
    T = fkine(q)
    print('Coordenadas obtenidas:\n', np.round(T,3))

    # Red marker shows the achieved position
    bmarker.xyz(T[0:3,3])
    # Green marker shows the desired position
    bmarker_des.xyz(xd)

    # Objeto (mensaje) de tipo JointState
    jstate = JointState()
    # Asignar valores al mensaje
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
        bmarker_des.publish()
        # Wait for the next iteration
        rate.sleep()

        plt.figure()
        plt.plot(e, 'b', label='Error')
        plt.plot(e, 'b.')
        plt.title("Evolución del error")
        plt.grid()
        plt.xlabel("Número de iteraciones")
        plt.ylabel("Norma del error")
        plt.legend()
        plt.savefig("error_evolution_k3.png", dpi=300)  # Save with high resolution

