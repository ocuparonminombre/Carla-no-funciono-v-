import sys
import os
import numpy as np
import cv2
import csv
from collections import deque

# --- CONFIGURACIÓN WEBOTS ---
WEBOTS_HOME = r"C:\Program Files\Webots"
sys.path.append(os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python'))
sys.path.append(os.path.join(WEBOTS_HOME, 'projects', 'default', 'libraries', 'vehicle', 'python'))
if sys.platform == "win32":
    os.add_dll_directory(os.path.join(WEBOTS_HOME, 'lib', 'controller'))

try:
    from vehicle import Driver
except ImportError:
    sys.exit("ERROR: No se encuentra Webots.")

# --- IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from vision.lane_detector import LaneDetector
from rl.dqn_agent import DQNAgent

# --- FUNCIONES ---
def calculate_reward(error, speed):
    if abs(error) > 0.9: return -10.0 # Castigo Choque
    reward = 0.05 # Sobrevivir
    if abs(error) < 0.1: reward += 1.0 # Centro perfecto
    elif abs(error) < 0.2: reward += 0.5
    return reward

def preprocess_for_ai(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (80, 80))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

def stack_frames(stacked_frames, frame, is_new_episode):
    if is_new_episode:
        stacked_frames = deque([frame] * 4, maxlen=4)
    else:
        stacked_frames.append(frame)
    return np.concatenate(stacked_frames, axis=0), stacked_frames

# --- BUCLE PRINCIPAL ---
def main():
    print("--- INICIANDO SISTEMA RESTAURADO ---")
    
    # Rutas Absolutas
    csv_path = os.path.join(current_dir, "historial_entrenamiento.csv")
    model_path = os.path.join(current_dir, "modelo_entrenado.pth")

    cv2.namedWindow("Conductor Autonomo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Conductor Autonomo", 800, 600)
    cv2.namedWindow("Ojo IA (Stack)", cv2.WINDOW_NORMAL)

    driver = Driver()
    time_step = int(driver.getBasicTimeStep())
    driver.setSteeringAngle(0.0)
    driver.setCruisingSpeed(0.0)
    
    camera = driver.getDevice("camera_front")
    camera.enable(time_step)
    
    vision = LaneDetector()
    agent = DQNAgent()
    
    # Cargar Cerebro
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(">> Cerebro cargado.")
        except:
            print(">> Cerebro nuevo.")

    # Crear CSV
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["Episodio", "RecompensaTotal", "Pasos", "Epsilon"])

    # Variables
    stacked_frames = deque(maxlen=4)
    prev_state = None      
    prev_action = 0        
    episode_reward = 0
    steps = 0
    episode_count = 0
    stuck_counter = 0

    print(">>> ENTRENANDO... (Se cerrará al chocar)")

    while driver.step() != -1:
        img_data = camera.getImage()
        if img_data:
            # 1. Visión
            img_raw = np.frombuffer(img_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            img_rgb = img_raw[:, :, :3]
            
            visual_img, lane_error = vision.process_image(img_rgb)
            frame_ai = preprocess_for_ai(visual_img)
            
            # 2. Estado
            is_new = (prev_state is None)
            current_state, stacked_frames = stack_frames(stacked_frames, frame_ai, is_new)

            # 3. Lógica de Fallo
            # Detector de atasco
            if abs(driver.getCurrentSpeed()) < 0.5: stuck_counter += 1
            else: stuck_counter = 0
            
            done = abs(lane_error) > 0.9 or stuck_counter > 150
            reward = calculate_reward(lane_error, 15)
            if done: reward = -10.0

            # 4. Entrenar
            if prev_state is not None:
                agent.remember(prev_state, prev_action, reward, current_state, done)
                loss = agent.train()
                episode_reward += reward

            # 5. Actuar
            action = agent.get_action(current_state)
            steering = 0
            if action == 1: steering = -0.3
            elif action == 2: steering = 0.3
            
            driver.setSteeringAngle(steering)
            driver.setCruisingSpeed(15)

            prev_state = current_state
            prev_action = action
            steps += 1
            
            # 6. CIERRE
            if done:
                print(f"--- Fin Episodio. Reward: {episode_reward:.2f} ---")
                agent.save(model_path)
                
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([episode_count, episode_reward, steps, agent.epsilon])
                
                print("🛑 ¡CHOQUE! Reinicia Webots (Ctrl+Shift+R) y vuelve a ejecutar.")
                cv2.destroyAllWindows()
                sys.exit()

            # 7. Visualización
            cv2.imshow("Conductor Autonomo", visual_img)
            ai_view = cv2.resize(current_state[-1], (300, 300), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Ojo IA (Stack)", ai_view)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    agent.save(model_path)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()