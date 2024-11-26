#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState

from markers import *
from functions_dh import *

if __name__ == '__main__':
    # Initialize the node
    rospy.init_node("testKineControlPose")
    print('Starting motion ... ')
    
    # Publisher: publish to the joint_states topic
    pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
    
    # Markers for the current and desired positions
    bmarker_current = FrameMarker()
    bmarker_desired = FrameMarker(0.5)
    
    # Joint names
    jnames = ['Slider_1', 'Revolute_2', 'Revolute_3', 'Slider_4', 'Revolute_6', 'Revolute_1']
    
    # Desired pose
    Rd = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    qd = Quaternion(matrix=Rd)
    # Find an xd that the robot can reach
    xd = np.array([1.5, 0, 1.80, qd.w, qd.x, qd.y, qd.z])
    
    # Initial configuration
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Resulting initial pose (end effector with respect to the base link)
    T = fkine(q0)
    x0 = TF2xyzquat(T)
    
    # Markers for the current and the desired pose
    #bmarker_current.setPose(x0)
    #bmarker_desired.setPose(xd)
    
    # Instance of the JointState message
    jstate = JointState()
    # Values of the message
    jstate.header.stamp = rospy.Time.now()
    jstate.name = jnames
    # Add the head joint value (with value 0) to the joints
    jstate.position = q0
    
    # Frequency (in Hz) and control period 
    freq = 50
    dt = 1.0 / freq
    rate = rospy.Rate(freq)
    
    # Initial joint configuration
    q = copy(q0)
    x = copy(x0)
    quat = x[3:7]
    # Initialize the derror vector (derivative of the error)
    derror = np.zeros(7)

    k = 0.1
    epsilon = 1e-4
    t = 0
    cnt = 0

    # Joint limits for q[0] and q[3]
    q0_limits = [0.0, 0.50]  # Example: Slider_1 limits
    q3_limits = [0.0, 0.35]  # Example: Slider_4 limits

    # Main loop
    while not rospy.is_shutdown():
        # Current time (needed for ROS)
        jstate.header.stamp = rospy.Time.now()
        
        # Calculate the error between poses
        J = jacobian_pose(q)
        T = fkine(q)
        x = TF2xyzquat(T)
        err_pose = PoseError(x, xd)
        if np.linalg.norm(err_pose) < epsilon:
            print("Se llegó al punto deseado en {:.3} segundos".format(cnt * dt))
            break
        
        de = -k * err_pose

        # Bloque para calcular el Jacobiano
        try:
            # Intentar calcular con Moore-Penrose (pseudo-inversa)
            dq = np.linalg.pinv(J).dot(de)
        except np.linalg.LinAlgError:
            # Si falla, usar el Jacobiano amortiguado
            damping_factor = 0.01
            J_damped = J.T.dot(np.linalg.inv(J.dot(J.T) + damping_factor * np.eye(J.shape[0])))
            dq = J_damped.dot(de)
        
        # Actualizar las configuraciones articulares
        q = q + dt * dq
        if q[0] < q0_limits[0]:
            q[0]= q[0]
        elif q[0] > q0_limits[1]:
            q[0] = q0_limits[1]

        if q[3] < q3_limits[0]:
            q[3] = q[0]
        elif q[3] > q3_limits[1]:
            q[3] = q3_limits[1]
        #q[0] = np.clip(q[0], q0_limits[0], q0_limits[1])
        #q[3] = np.clip(q[3], q3_limits[0], q3_limits[1])

        # Evitar un bucle infinito
        cnt += 1
        t += 1
        if cnt > 1e5:
            print("Se excedió el número de iteraciones")
            break
        
        # Actualizar la transformación actual
        T = fkine(q)
        x = TF2xyzquat(T)

        # Publicar el mensaje
        jstate.position = q
        pub.publish(jstate)
        #bmarker_desired.setPose(xd)
        #bmarker_current.setPose(x)
        
        # Esperar para la siguiente iteración
        rate.sleep()
