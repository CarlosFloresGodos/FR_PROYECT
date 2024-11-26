#!/usr/bin/env python3
import numpy as np
from copy import copy
from pyquaternion import Quaternion
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import rbdl

pi = np.pi


class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('../urdf/robot_er.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq


def dh(d, theta, a, alpha):
 """
 Calcular la matriz de transformacion homogenea asociada con los parametros
 de Denavit-Hartenberg.
 Los valores d, theta, a, alpha son escalares.
 """
 sth = np.sin(theta)
 cth = np.cos(theta)
 sa  = np.sin(alpha)
 ca  = np.cos(alpha)
 T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
               [sth,  ca*cth, -sa*cth, a*sth],
               [0.0,      sa,      ca,     d],
               [0.0,     0.0,     0.0,   1.0]])
 return T



def fkine(q):
 """
 Calcular la cinematica directa del brazo robotico dados sus valores articulares. 
 q es un vector numpy de la forma [q1, q2, q3, ..., qn]
 """
 # Matrices DH (completar)
 T1 = dh(0,q[5],0,0)
 T2 = dh(q[0]+0.95,-pi/2,0,pi/2)
 T3 = dh(0.433, pi+q[1], 0, pi/2)
 T4 = dh(0.839, 0+q[2], 0, pi/2)
 T5 = dh(1.03+q[3], 0, 0, 0)
 T6 = dh(0.0, 0+q[4], 0, 0)
 T = T1@T2@T3@T4@T5@T6
 return T


def rot2quat(R):
 """
 Convertir una matriz de rotacion en un cuaternion

 Entrada:
  R -- Matriz de rotacion
 Salida:
  Q -- Cuaternion [ew, ex, ey, ez]

 """
 dEpsilon = 1e-6
 quat = 4*[0.,]

 quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
 if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
  quat[1] = 0.0
 else:
  quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
 if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
  quat[2] = 0.0
 else:
  quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
 if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
  quat[3] = 0.0
 else:
  quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

 return np.array(quat)


def TF2xyzquat(T):
 """
 Convert a homogeneous transformation matrix into the a vector containing the
 pose of the robot.

 Input:
  T -- A homogeneous transformation
 Output:
  X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
       is Cartesian coordinates and the last part is a quaternion
 """
 quat = rot2quat(T[0:3,0:3])
 res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
 return np.array(res)


def jacobian(q, delta=0.0001):
 """
 Jacobiano analitico para la posicion de un brazo robotico de n grados de libertad. 
 Retorna una matriz de 3xn y toma como entrada el vector de configuracion articular 
 q=[q1, q2, q3, ..., qn]
 """
 # Crear una matriz 3xn
 n = q.size
 J = np.zeros((3,n))
 # Calcular la transformacion homogenea inicial (usando q)
 T = fkine(q)
    
 # Iteracion para la derivada de cada articulacion (columna)
 for i in range(n):
  # Copiar la configuracion articular inicial
  dq = copy(q)
  # Calcular nuevamenta la transformacion homogenea e
  # Incrementar la articulacion i-esima usando un delta
  dq[i] += delta
  # Transformacion homogenea luego del incremento (q+delta)
  T_inc = fkine(dq)
  # Aproximacion del Jacobiano de posicion usando diferencias finitas
  J[0:3,i]=(T_inc[0:3,3]-T[0:3,3])/delta
 return J


def ikine(xdes, q0):
 """
 Calcular la cinematica inversa de un brazo robotico numericamente a partir 
 de la configuracion articular inicial de q0. Emplear el metodo de newton.
 """
 epsilon  = 0.001
 max_iter = 1000
 ee = []
 q_limits = [(0, 0.50), None, None, (0,0.35), None, None]  # Límites para q[0] y q[5]
 #delta    = 0.00001

 q  = copy(q0)

 # Bucle principal
 epsilon = 0.001
 max_iter = 10000
 ee = []  # Almacena el error en cada iteración
 q = copy(q0)

# Bucle principal
 for i in range(max_iter):
  J = jacobian(q)  # Calcula el Jacobiano
  f = fkine(q)[0:3, 3]  # Posición actual del efector final
  e = xdes - f  # Error en el espacio cartesiano

    # Actualización de q usando el método de Newton
  q = q + np.dot(np.linalg.pinv(J), e)

    # Aplicar límites a las articulaciones
  if q_limits is not None:
   for j in range(len(q)):
    if q_limits[j] is not None:
     q[j] = np.clip(q[j], q_limits[j][0], q_limits[j][1])

    # Calcular norma del error
  enorm = np.linalg.norm(e)
  ee.append(enorm)
    # Condición de término
  if enorm < epsilon:
   break
  if i == max_iter - 1:
   print("El algoritmo no llegó al valor deseado")
 print(f"Iteración {i}, Error: {np.round(enorm, 4)}")
 return q, ee
def ik_gradient(xdes, q0):
 """
 Calcular la cinematica inversa de un brazo robotico numericamente a partir 
 de la configuracion articular inicial de q0. Emplear el metodo gradiente.
 """
 q_limits = [(0, 0.50), None, None, (0,0.35), None, None]  # Límites para q[0] y q[5]
 epsilon  = 0.001
 max_iter = 10000
 delta    = 0.01

 q  = copy(q0)
 ee = []  # Almacena el error en cada iteración

# Bucle principal
 for i in range(max_iter):
  J = jacobian(q)  # Calcula el Jacobiano
  f = fkine(q)[0:3, 3]  # Posición actual del efector final
  e = xdes - f  # Error en el espacio cartesiano
    # Actualización del valor articular (descenso del gradiente)
  q = q + delta*np.dot(J.T, e)
    # Aplicar límites a las articulaciones
  if q_limits is not None:
   for j in range(len(q)):
    if q_limits[j] is not None:
     q[j] = np.clip(q[j], q_limits[j][0], q_limits[j][1])

    # Calcular norma del error
  enorm = np.linalg.norm(e)
  ee.append(enorm)
    # Condición de término
  if enorm < epsilon:
   break
  if i == max_iter - 1:
   print("El algoritmo no llegó al valor deseado")
 print(f"Iteración {i}, Error: {np.round(enorm, 4)}")
 return q, ee

def jacobian_pose(q, delta=0.0001):
 """
 Jacobiano analitico para la posicion y orientacion (usando un
 cuaternion). Retorna una matriz de 7xn y toma como entrada el vector de
 configuracion articular q=[q1, q2, q3, ..., qn]
 """
 n = q.size
 J = np.zeros((7, n))

# Calcular la transformación homogénea inicial y la pose inicial
 T = fkine(q)
 pose = TF2xyzquat(T)

# Iteración para la derivada de cada articulación (columna)
 for i in range(n):
    # Copiar la configuración articular inicial
  dq = copy(q)
    # Incrementar la articulación i-ésima usando un delta
  dq[i] += delta
    # Calcular la nueva transformación y la pose incrementada
  T_inc = fkine(dq)
  pose_inc = TF2xyzquat(T_inc)
    
    # Aproximación del Jacobiano de posición usando diferencias finitas
  J[0:3, i] = (pose_inc[0:3] - pose[0:3]) / delta  # Posición x, y, z
    # Aproximación del Jacobiano de orientación usando diferencias finitas
  J[3:7, i] = (pose_inc[3:7] - pose[3:7]) / delta  # Cuaterniones ew, ex, ey, ez
 return J

def PoseError(x,xd):
 """
 Determine the pose error of the end effector.

 Input:
 x -- Actual position of the end effector, in the format [x y z ew ex ey ez]
 xd -- Desire position of the end effector, in the format [x y z ew ex ey ez]
 Output:
 err_pose -- Error position of the end effector, in the format [x y z ew ex ey ez]
 """
 pos_err = x[0:3]-xd[0:3]
 qact = Quaternion(x[3:7])
 qdes = Quaternion(xd[3:7])
 qdif =  qdes*qact.inverse
 qua_err = np.array([qdif.w,qdif.x,qdif.y,qdif.z])
 err_pose = np.hstack((pos_err,qua_err))
 
 return err_pose

def show_graph(k):
    # Cargar los datos desde los archivos
    posiciones_deseadas = np.loadtxt('/home/user/xdesired.txt')
    posiciones_actuales = np.loadtxt('/home/user/xcurrent.txt')

    # Dividir los datos en variables
    tiempo_deseadas = posiciones_deseadas[:, 0]
    deseada_x = posiciones_deseadas[:, 1]
    deseada_y = posiciones_deseadas[:, 2]
    deseada_z = posiciones_deseadas[:, 3]

    tiempo_actuales = posiciones_actuales[:, 0]
    actual_x = posiciones_actuales[:, 1]
    actual_y = posiciones_actuales[:, 2]
    actual_z = posiciones_actuales[:, 3]

    # Graficar la posición deseada y la posición actual como función del tiempo
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, 1ª gráfica
    plt.plot(tiempo_deseadas, deseada_x, label='Deseada X', linestyle='--', linewidth=2)
    plt.plot(tiempo_actuales, actual_x, label='Actual X', linestyle='-', linewidth=2)
    plt.plot(tiempo_deseadas, deseada_y, label='Deseada Y', linestyle='--', linewidth=2)
    plt.plot(tiempo_actuales, actual_y, label='Actual Y', linestyle='-', linewidth=2)
    plt.plot(tiempo_deseadas, deseada_z, label='Deseada Z', linestyle='--', linewidth=2)
    plt.plot(tiempo_actuales, actual_z, label='Actual Z', linestyle='-', linewidth=2)
    plt.xlabel('Tiempo')
    plt.ylabel('Posición')
    plt.title(f'Posición Deseada y Actual vs. Tiempo, k{k}')
    plt.legend()
    plt.grid()

    # Graficar la posición deseada y actual en el espacio cartesiano (x, y, z)
    ax = plt.subplot(1, 2, 2, projection='3d')  # 1 fila, 2 columnas, 2ª gráfica en 3D
    # Línea para la trayectoria deseada
    ax.plot(deseada_x, deseada_y, deseada_z, color='b', label='Trayectoria Deseada', linewidth=1)
    # Línea para la trayectoria actual
    ax.plot(actual_x, actual_y, actual_z, color='r', label='Trayectoria Actual', linewidth=1)
    # Punto inicial y final de la trayectoria actual
    ax.scatter(actual_x[0], actual_y[0], actual_z[0], color='g', marker='o', s=50, label='Posición Inicial')
    ax.scatter(actual_x[-1], actual_y[-1], actual_z[-1], color='r', marker='o', s=50, label='Posición Final')

    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    ax.set_title('Posición Deseada y Actual en 3D')
    ax.set_xlim([-2, 2])  # Límites para el eje X
    ax.set_ylim([-2, 2])  # Límites para el eje Y
    ax.set_zlim([-2, 2])  # Límites para el eje Z
    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # Escala uniforme en los ejes

    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()