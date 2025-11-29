import sys
import os
import numpy as np
import cv2
import csv

# --- 1. CONFIGURACIÓN WEBOTS ---
WEBOTS_HOME = r"C:\Program Files\Webots"
sys.path.append(os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python'))
sys.path.append(os.path.join(WEBOTS_HOME, 'projects', 'default', 'libraries', 'vehicle', 'python'))
if sys.platform == "win32":
    os.add_dll_directory(os.path.join(WEBOTS_HOME, 'lib', 'controller'))

try:
    from vehicle import Driver
except ImportError:
    sys.exit("ERROR: No se encuentra Webots.")

# --- 2. IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from vision.lane_detector import LaneDetector
from rl.dqn_agent import DQNAgent

def calculate_reward(error, speed):
    # Castigo fuerte si se sale
    if abs(error) > 0.3:
        return -5.0
    # Recompensa normal
    reward = 0.1
    # Bonus por ir centrado
    if abs(error) < 0.1:
        reward += 1.0
    return reward

def preprocess_for_ai(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (80, 80))
    normalized = resized / 255.0
    input_ready = np.expand_dims(normalized, axis=0)
    return input_ready

# --- 3. BUCLE PRINCIPAL ---
def main():
    print("--- LISTO PARA RODAR ---")
    
    # Configuración de Ventanas Grandes
    cv2.namedWindow("Conductor Autonomo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Conductor Autonomo", 800, 600)
    cv2.namedWindow("Ojo IA Gigante", cv2.WINDOW_NORMAL)

    # Instanciamos SOLAMENTE el Driver (Quitamos Supervisor para evitar error)
    driver = Driver()
    time_step = int(driver.getBasicTimeStep())
    driver.setSteeringAngle(0.0)
    driver.setCruisingSpeed(0.0)
    
    camera = driver.getDevice("camera_front")
    camera.enable(time_step)
    
    vision = LaneDetector()
    agent = DQNAgent()
    
    if os.path.exists("modelo_entrenado.pth"):
        agent.load("modelo_entrenado.pth")

    # Variables
    state_current = None
    episode_reward = 0
    steps = 0
    episode_count = 0
    stuck_counter = 0

    while driver.step() != -1:
        img_data = camera.getImage()
        if img_data:
            # Procesar
            img_raw = np.frombuffer(img_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            img_rgb = img_raw[:, :, :3]
            
            # Visión y Decisión
            visual_img, lane_error = vision.process_image(img_rgb)
            ai_input = preprocess_for_ai(visual_img)
            
            if state_current is None: state_current = ai_input

            action = agent.get_action(state_current)
            
            # Actuar
            steering = 0
            if action == 1: steering = -0.3
            elif action == 2: steering = 0.3
            
            driver.setSteeringAngle(steering)
            driver.setCruisingSpeed(15)
            
            # Detectar Atascos (Velocidad 0)
            current_speed = driver.getCurrentSpeed()
            if abs(current_speed) < 1.0: stuck_counter += 1
            else: stuck_counter = 0
            
            # Recompensas
            reward = calculate_reward(lane_error, 15)
            
            done_lane = abs(lane_error) > 0.4
            done_stuck = stuck_counter > 100
            done = done_lane or done_stuck
            
            if done:
                reward = -10
                if done_lane: print(f"Episodio {episode_count}: SALIDA DE CARRIL")
                if done_stuck: print(f"Episodio {episode_count}: ATASCADO")
            
            episode_reward += reward
            
            # Entrenar
            state_next = ai_input
            agent.remember(state_current, action, reward, state_next, done)
            loss = agent.train()
            
            state_current = state_next
            steps += 1
            
            # LÓGICA DE REINICIO MANUAL
            if done:
                # Guardamos cerebro
                if episode_count % 5 == 0:
                    agent.save("modelo_entrenado.pth")
                
                print(f">>> FIN EPISODIO. Reinicia variables...")
                
                # Reseteamos solo las variables de memoria
                state_current = None
                episode_reward = 0
                steps = 0
                stuck_counter = 0
                episode_count += 1
                
                # NOTA: Como quitamos el Supervisor, el auto NO volverá al inicio solo.
                # Si se quedó trabado, presiona Ctrl+Shift+R en Webots tú mismo.
                # Si solo se salió un poco, intentará seguir desde ahí.

            # Visualización
            cv2.imshow("Conductor Autonomo", visual_img)
            ai_big = cv2.resize(ai_input[0], (400, 400), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Ojo IA Gigante", ai_big)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    agent.save("modelo_entrenado.pth")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()