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
    rospy.init_node("control_pdg")
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
    qdes = np.array([0.2, -0.3, 0.2, 0.3, 0.2, -pi])
    xdes = fkine(qdes)[0:3, 3]
    #print(xdes)
    jstate.position = q
    pub.publish(jstate)

    # Modelo RBDL
    modelo = rbdl.loadModel('../urdf/robot_er.urdf')
    ndof = modelo.q_size

    freq = 20
    dt = 1.0 / freq
    rate = rospy.Rate(freq)

    # Definir los límites para q[0] y q[3]
    q0_min, q0_max = -1.0, 1.0  # Ejemplo de límites para q[0]
    q3_min, q3_max = 0.0, 0.5   # Ejemplo de límites para q[3]      

    robot = Robot(q, dq, ndof, dt)

    # Ganancias del controlador
    Kp = np.diag([.4, .2, 0.3, .2, 2, .1])
    Kd = np.diag([.3, .5, .6, .1, .2, .1])

    # Listas para almacenar datos
    time_data = []
    x_data, y_data, z_data = [], [], []

    epsilon = 0.01  # Tolerancia para la convergencia
    max_time = 40.0  # Tiempo mínimo de ejecución en segundos
    t = 0.0
    q0_limits = [0.0, 0.50]  # Example: Slider_1 limits
    q3_limits = [0.0, 0.35]  # Example: Slider_4 limits
    while not rospy.is_shutdown():
        # Leer estado actual
        q = robot.read_joint_positions()
        dq = robot.read_joint_velocities()
        x = fkine(q)[0:3, 3]

        #q[0] = np.clip(q[0], q0_min, q0_max)
        #q[3] = np.clip(q[3], q3_min, q3_max)
        if q[0] < q0_limits[0]:
            q[0]= q0_limits[0]
            print("min1")
        elif q[0] > q0_limits[1]:
            q[0] = q0_limits[1]
            print("max1")

        if q[3] < q3_limits[0]:
            q[3] = q3_limits[0]
            print("min2")
        elif q[3] > q3_limits[1]:
            q[3] = q3_limits[1]
            print("max2")
        # Comprobar convergencia después de n segundos
        if t >= max_time and np.linalg.norm(x - xdes) < epsilon:
            rospy.loginfo("El efector final ha alcanzado la posición deseada")
            break

        # Actualizar datos para graficar
        time_data.append(t)
        x_data.append(x[0])
        y_data.append(x[1])
        z_data.append(x[2])

        # Guardar datos en archivos
        fxact.write(f"{t} {x[0]} {x[1]} {x[2]}\n")
        fxdes.write(f"{t} {xdes[0]} {xdes[1]} {xdes[2]}\n")
        fqact.write(f"{t} {q[0]} {q[1]} {q[2]} {q[3]} {q[4]} {q[5]}\n")
        fqdes.write(f"{t} {qdes[0]} {qdes[1]} {qdes[2]} {qdes[3]} {qdes[4]} {qdes[5]}\n")

        # Control dinámico
        g = np.zeros(ndof)
        rbdl.InverseDynamics(modelo, q, np.zeros(ndof), np.zeros(ndof), g)
        u = Kp.dot(qdes - q) + Kd.dot(-dq) + g

        # Simulación
        robot.send_command(u)

        # Publicar estado
        jstate.header.stamp = rospy.Time.now()
        jstate.position = q
        pub.publish(jstate)

        # Actualizar marcadores
        bmarker_deseado.xyz(xdes)
        bmarker_actual.xyz(x)

        t += dt
        rate.sleep()

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
        plt.savefig("grafica_evolucion.png")
        rospy.loginfo("Gráfica guardada")
    except Exception as e:
        rospy.logerr(f"Error al generar la gráfica: {e}")
