#activar linea 87 a 93 para cargar modelo entrenado
"""
SISTEMA DE CONDUCCIÓN AUTÓNOMA INTEGRADO (FINAL)
Integración: Webots + OpenCV + PyTorch (DQN)
"""
import sys
import os
import numpy as np
import cv2
import torch

# --- 1. CONFIGURACIÓN DE RUTAS (FIX WINDOWS) ---
WEBOTS_HOME = r"C:\Program Files\Webots"
sys.path.append(os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python'))
sys.path.append(os.path.join(WEBOTS_HOME, 'projects', 'default', 'libraries', 'vehicle', 'python'))
if sys.platform == "win32":
    os.add_dll_directory(os.path.join(WEBOTS_HOME, 'lib', 'controller'))

try:
    from vehicle import Driver
except ImportError:
    sys.exit("ERROR CRITICO: No se encuentra Webots.")

# --- 2. IMPORTAR MÓDULOS DEL EQUIPO ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vision.lane_detector import LaneDetector
from rl.dqn_agent import DQNAgent

# --- 3. FUNCIONES AUXILIARES ---
# --- EN MAIN_DRIVER.PY ---

def preprocess_for_ai(image):
    """Convierte la imagen visual a lo que entiende la Red Neuronal"""
    # 1. Escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. Reducir tamaño a 80x80
    resized = cv2.resize(gray, (80, 80))
    # 3. Normalizar (0 a 1)
    normalized = resized / 255.0
    
    # --- FIX CRITICO: AGREGAR DIMENSIÓN DE CANAL ---
    # Convertimos de (80, 80) a (1, 80, 80)
    # Esto le dice a PyTorch: "Es 1 solo canal de color (blanco y negro)"
    input_ready = np.expand_dims(normalized, axis=0)
    
    return input_ready

def calculate_reward(error, speed):
    """
    DEFINE EL COMPORTAMIENTO:
    - Queremos error 0 (centro del carril).
    - Queremos velocidad constante.
    """
    # Recompensa base por estar vivo
    reward = 0.5
    
    # Penalización por desviarse del centro
    # El error va de -1 a 1. Si es 0, penalización es 0. Si es 1, penalización es alta.
    penalty = abs(error) * 2.0
    
    reward -= penalty
    
    # Bonificación extra si está muy centrado
    if abs(error) < 0.1:
        reward += 1.0
        
    return reward

# --- 4. BUCLE PRINCIPAL ---
def main():
    print("--- INICIANDO SISTEMA DE IA ---")
    
    # Configuración Webots
    driver = Driver()
    time_step = int(driver.getBasicTimeStep())
    driver.setSteeringAngle(0.0)
    driver.setCruisingSpeed(0.0)
    
    camera = driver.getDevice("camera_front")
    camera.enable(time_step)
    
    # Instanciar IA
    vision = LaneDetector()
    agent = DQNAgent() # Cerebro cargado
    
    # CARGAR CEREBRO PREVIO activar cuando ya este entrenado
    #if os.path.exists("modelo_entrenado.pth"):
     #   agent.load("modelo_entrenado.pth")
      #  print("!!! CEREBRO CARGADO - Modo Experto !!!")
    #else:
     #   print("No hay cerebro guardado, empezando desde cero.")
    
    
    
    # Variables de estado
    state_current = None
    state_next = None
    episode_reward = 0
    steps = 0
    
    print("--- ENTRENAMIENTO INICIADO ---")
    print("El auto empezará a explorar. ¡Ten paciencia!")

    while driver.step() != -1:
        # A. OBTENER IMAGEN
        img_data = camera.getImage()
        if img_data:
            # Procesar imagen cruda
            img_raw = np.frombuffer(img_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            img_rgb = img_raw[:, :, :3]
            
            # B. VISIÓN (Jhessica)
            # Obtenemos la imagen procesada y el error numérico (-1 a 1)
            visual_img, lane_error = vision.process_image(img_rgb)
            
            # C. PREPARAR ESTADO PARA IA
            # Usamos la imagen procesada (con las lineas verdes) porque ayuda a la IA
            ai_input = preprocess_for_ai(visual_img)
            
            # D. DECISIÓN (Nicolas)
            # Si es el primer paso, inicializamos
            if state_current is None:
                state_current = ai_input
            
            # El agente decide qué hacer basado en la imagen actual
            action = agent.get_action(state_current)
            
            # E. ACTUAR EN EL MUNDO
            # Mapear acción (0,1,2) a volante
            steering = 0
            speed = 15 # Km/h constantes
            
            if action == 0:   # Recto
                steering = 0
            elif action == 1: # Izquierda
                steering = -0.3
            elif action == 2: # Derecha
                steering = 0.3
            
            driver.setSteeringAngle(steering)
            driver.setCruisingSpeed(speed)
            
            # F. CALCULAR RECOMPENSA (Consecuencias)
            reward = calculate_reward(lane_error, speed)
            episode_reward += reward
            
            # G. GUARDAR EXPERIENCIA Y ENTRENAR
            state_next = ai_input
            
            # Definir si "perdió" (Done)
            # Si el error es muy grande (>0.8), asumimos que se salió del carril
            done = abs(lane_error) > 0.8
            
            if done:
                reward = -10 # Castigo fuerte por salirse
                print(f"¡SALIDA DE CARRIL! Reward: {reward}")
            
            # Guardar en memoria: (Estado, Acción, Premio, Nuevo Estado, ¿Perdió?)
            agent.remember(state_current, action, reward, state_next, done)
            
            # ¡APRENDER! (Entrena la red neuronal)
            loss = agent.train()
            
            # ... (después de loss = agent.train()) ...

            # --- GUARDADO AUTOMÁTICO (CHECKPOINT) ---
            # Guardar cada 500 pasos (aprox cada 2-3 minutos)
            if steps % 500 == 0:
                print(">>> GUARDANDO CEREBRO (CHECKPOINT) <<<")
                agent.save("modelo_entrenado.pth")
            
            # Actualizar estado para el siguiente frame
            state_current = state_next
            steps += 1
            
            # H. VISUALIZACIÓN
            # Mostrar lo que ve la IA (pequeño)
            cv2.imshow("Ojo de la IA (80x80)", cv2.resize(ai_input[0], (200, 200)))
            
            # Mostrar visión humana con datos
            info_text = f"Action: {action} | Error: {lane_error:.2f}"
            cv2.putText(visual_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if loss > 0:
                cv2.putText(visual_img, f"Learning Loss: {loss:.4f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            cv2.imshow("Conductor Autonomo", visual_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # NOTA: Como no usamos Supervisor, el reinicio es manual o cíclico.
            # Si se sale mucho, podrías frenar el auto temporalmente.

    # Guardar el cerebro al cerrar
    agent.save("modelo_entrenado.pth")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()