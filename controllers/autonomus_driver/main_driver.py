#script principal que ejecuta webots para el conductor autónomo
"""
Controlador Principal - Sistema de Conducción Autónoma
Integración de Visión y RL en Webots
"""
from vehicle import Driver  # Librería nativa de Webots para coches
from vision.lane_detector import LaneDetector
from rl.dqn_agent import DQNAgent

# Configuración de tiempo (Time step del simulador)
TIME_STEP = 32

def main():
    # 1. Inicializar el vehículo (Webots)
    driver = Driver()
    driver.setSteeringAngle(0.0)
    driver.setCruisingSpeed(0.0)
    
    # 2. Inicializar sensores
    # Nota: Los nombres 'camera_front' deben coincidir con lo que Carlos ponga en el mundo
    camera = driver.getDevice("camera_front")
    camera.enable(TIME_STEP)
    
    # gps = driver.getDevice("gps") ...
    
    # 3. Instanciar módulos del equipo
    vision_system = LaneDetector()  # Módulo de Jhessica
    rl_brain = DQNAgent()           # Módulo de Nicolas
    
    print("Sistema de Conducción Autónoma Iniciado...")

    # Bucle principal de simulación
    while driver.step() != -1:
        # A. Obtener datos sensoriales
        # image = camera.getImage()
        
        # B. Procesar visión
        # lane_info = vision_system.process(image)
        
        # C. Tomar decisión (RL)
        # action = rl_brain.choose_action(lane_info)
        
        # D. Ejecutar acción en el motor
        # driver.setSteeringAngle(action['steering'])
        # driver.setCruisingSpeed(action['speed'])
        
        pass

if __name__ == "__main__":
    main()