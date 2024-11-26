import rbdl
import numpy as np

if __name__ == '__main__':
  
  # Lectura del modelo del robot a partir de URDF (parsing)
  modelo = rbdl.loadModel('../urdf/robot_er.urdf')
  # Grados de libertad
  ndof = modelo.q_size

  # Configuracion articular
  q = np.array([0.5, 0.2, 0.3, 0.8, 0.5, 0.6])
  # Velocidad articular
  dq = np.array([0.8, 0.7, 0.8, 0.6, 0.9, 1.0])
  # Aceleracion articular
  ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5])
  
  # Arrays numpy
  zeros = np.zeros(ndof)          # Vector de ceros
  tau   = np.zeros(ndof)          # Para torque
  g     = np.zeros(ndof)          # Para la gravedad
  c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
  M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
  e     = np.eye(6)               # Vector identidad
  
  # Torque dada la configuracion del robot
  rbdl.InverseDynamics(modelo, q, dq, ddq, tau)
  
  # Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga,
  # y matriz M usando solamente InverseDynamics
  # Vector de gravedad
  rbdl.InverseDynamics(modelo, q, zeros, zeros, g)

  # Vector de Coriolis y centrífuga
  rbdl.InverseDynamics(modelo, q, dq, zeros, c)
  c -= g  # Eliminamos la contribución gravitatoria

  # Matriz de inercia
  for i in range(ndof):
      ddq_unit = np.zeros(ndof)
      ddq_unit[i] = 1.0  # Aceleración unitaria en la i-ésima articulación
      tau_temp = np.zeros(ndof)
      rbdl.InverseDynamics(modelo, q, zeros, ddq_unit, tau_temp)
      M[:, i] = tau_temp - g  # Restamos el componente gravitacional

  # Mostramos los resultados de g, c y M redondeados a 3 decimales
  print("\nResultados usando InverseDynamics:")
  print("Vector de gravedad g(q):\n", np.round(g, 3))
  print("Vector de Coriolis c(q, dq):\n", np.round(c, 3))
  print("Matriz de inercia M(q):\n", np.round(M, 3))


  # Parte 2: Calcular M y los efectos no lineales b usando las funciones
  # CompositeRigidBodyAlgorithm y NonlinearEffects. Almacenar los resultados
  # en los arreglos llamados M2 y b2
  b2 = np.zeros(ndof)          # Para efectos no lineales
  M2 = np.zeros([ndof, ndof])  # Para matriz de inercia
  
  # Matriz de inercia usando CompositeRigidBodyAlgorithm
  rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)

  # Efectos no lineales usando NonlinearEffects
  rbdl.NonlinearEffects(modelo, q, dq, b2)

  # Mostramos los resultados de M2 y b2 redondeados a 3 decimales
  print("\nResultados usando funciones específicas:")
  print("Matriz de inercia M2 (CompositeRigidBodyAlgorithm):\n", np.round(M2, 3))
  print("Vector b2 (NonlinearEffects):\n", np.round(b2, 3))
  print("Vector c+g:\n", np.round(c+g, 3))
  
  
  # Parte 2: Verificacion de valores
  
  print("\nVerificación de valores:")
  print("Diferencia entre M y M2:", np.linalg.norm(M - M2))
  print("Diferencia entre b2 y (c + g):", np.linalg.norm(b2 - (c + g)))
  
  
  
  # Parte 3: Verificacion de la expresion de la dinamica
  tau_verificado = M @ ddq + c + g
  print("\nVerificación de la expresión dinámica:")
  print("Matriz Tau:\n", np.round(tau,3))
  print("Matriz Tau verificado:\n", np.round(tau_verificado,3)) 
  print("Diferencia entre tau calculado directamente y tau verificado:", np.linalg.norm(tau - tau_verificado))