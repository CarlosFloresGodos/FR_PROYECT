#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions_dh import *
from roslib import packages
import rbdl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    rospy.init_node("control_dinamica_inversa")
    pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
    bmarker_actual = BallMarker(color['RED'])
    bmarker_deseado = BallMarker(color['GREEN'])

    # Archivos para almacenar datos
    fqact = open("/home/user/qactual.txt", "w")
    fqdes = open("/home/user/qdeseado.txt", "w")
    fxact = open("/home/user/xactual.txt", "w")
    fxdes = open("/home/user/xdeseado.txt", "w")

    # Nombres de las articulaciones
    jnames = ['Slider_1', 'Revolute_2', 'Revolute_3', 'Slider_4', 'Revolute_6', 'Revolute_1']
    jstate = JointState()
    jstate.header.stamp = rospy.Time.now()
    jstate.name = jnames

    # Configuración inicial
    q = np.array([0.0, 0.0, 0, 0, 0, 0.0])
    dq = np.array([0., 0., 0., 0., 0., 0.])
    qdes = np.array([0.2, -0.3, 0.2, 0.3, 0.2, -np.pi])
    ddq = np.array([0.2, 0.3, 0.1, 0.2, 0.1, 0.3]) 
    xdes = fkine(qdes)[0:3, 3]
    jstate.position = q
    pub.publish(jstate)

    # Modelo RBDL
    modelo = rbdl.loadModel('../urdf/robot_er.urdf')
    ndof = modelo.q_size

    freq = 100
    dt = 1.0 / freq
    rate = rospy.Rate(freq)

    # Definir límites para q[0] y q[3]
    q0_limits = [0.0, 0.50]
    q3_limits = [0.0, 0.35]

    robot = Robot(q, dq, ndof, dt)

    # Listas para almacenar datos
    time_data = []
    x_data, y_data, z_data = [], [], []

    epsilon = 0.01  # Tolerancia para la convergencia
    max_time = 40.0  # Tiempo máximo de ejecución
    t = 0.0

    while not rospy.is_shutdown():
        # Leer estado actual
        q = robot.read_joint_positions()
        dq = robot.read_joint_velocities()
        x = fkine(q)[0:3, 3]

        # Restringir q[0] y q[3] dentro de los límites
        if q[0] < q0_limits[0]:
            q[0] = q0_limits[0]
        elif q[0] > q0_limits[1]:
            q[0] = q0_limits[1]

        if q[3] < q3_limits[0]:
            q[3] = q3_limits[0]
        elif q[3] > q3_limits[1]:
            q[3] = q3_limits[1]

        # Calcular dinámica inversa
        g = np.zeros(ndof)  # Gravedad
        zeros = np.zeros(ndof)  # Velocidades y aceleraciones nulas
        rbdl.InverseDynamics(modelo, q, zeros, zeros, g)

        

        # Matriz de inercia (M), fuerzas de Coriolis y gravedad (cg)
        Mq = np.zeros((ndof, ndof))
        rbdl.CompositeRigidBodyAlgorithm(modelo, q, Mq)

        # Fuerzas de Coriolis y centrífugas
        cg = np.zeros(ndof)
        rbdl.NonlinearEffects(modelo, q, dq, cg)

        # Ley de control: dinámica inversa
        u = Mq @ ddq + cg
        #print(f"Comando de control: {u}")

        # Simulación
        robot.send_command(u)

        # Guardar datos en archivos
        fxact.write(f"{t} {x[0]} {x[1]} {x[2]}\n")
        fxdes.write(f"{t} {xdes[0]} {xdes[1]} {xdes[2]}\n")
        fqact.write(f"{t} {q[0]} {q[1]} {q[2]} {q[3]} {q[4]} {q[5]}\n")
        fqdes.write(f"{t} {qdes[0]} {qdes[1]} {qdes[2]} {qdes[3]} {qdes[4]} {qdes[5]}\n")

        # Publicar estado
        jstate.header.stamp = rospy.Time.now()
        jstate.position = q
        pub.publish(jstate)

        # Actualizar marcadores
        bmarker_deseado.xyz(xdes)
        bmarker_actual.xyz(x)

        # Actualizar tiempo y dormir
        t += dt
        rate.sleep()

    # Cerrar archivos
    fqact.close()
    fqdes.close()
    fxact.close()
    fxdes.close()

    # Graficar resultados
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(time_data, x_data, label='x(t)', color='r')
        plt.plot(time_data, y_data, label='y(t)', color='g')
        plt.plot(time_data, z_data, label='z(t)', color='b')
        plt.axhline(y=xdes[0], color='r', linestyle='--', label='x_des')
        plt.axhline(y=xdes[1], color='g', linestyle='--', label='y_des')
        plt.axhline(y=xdes[2], color='b', linestyle='--', label='z_des')
        plt.title("Evolución temporal de las coordenadas del efector final")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Coordenadas (m)")
        plt.legend()
        plt.grid()

        # Guardar la gráfica
        plt.savefig("grafica_evolucion_sin_control.png")
        rospy.loginfo("Gráfica guardada")
    except Exception as e:
        rospy.logerr(f"Error al generar la gráfica: {e}")
