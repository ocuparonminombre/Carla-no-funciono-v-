
"""
CALIBRADOR V2 - EL ESPÍA DE COLORES
"""
import sys
import os
import cv2
import numpy as np

# --- CONFIGURACIÓN DE RUTAS ---
WEBOTS_HOME = r"C:\Program Files\Webots"
sys.path.append(os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python'))
sys.path.append(os.path.join(WEBOTS_HOME, 'projects', 'default', 'libraries', 'vehicle', 'python'))
if sys.platform == "win32":
    os.add_dll_directory(os.path.join(WEBOTS_HOME, 'lib', 'controller'))

from vehicle import Driver

# Variables globales para el mouse
pixel_hsv = None

def pick_color(event, x, y, flags, param):
    """ Función que se activa al hacer clic en la imagen """
    global pixel_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        # param es la imagen HSV
        pixel = param[y, x]
        pixel_hsv = pixel
        print(f"\n--- COLOR DETECTADO EN ({x},{y}) ---")
        print(f"H (Matiz):      {pixel[0]}")
        print(f"S (Saturación): {pixel[1]}")
        print(f"V (Brillo):     {pixel[2]}")
        print("-----------------------------------")
        print(f"SUGERENCIA PARA TU CÓDIGO:")
        print(f"lower = np.array([{max(0, pixel[0]-10)}, {max(0, pixel[1]-40)}, {max(0, pixel[2]-40)}])")
        print(f"upper = np.array([{min(179, pixel[0]+10)}, 255, 255])")

def main():
    driver = Driver()
    driver.setSteeringAngle(0.0)
    driver.setCruisingSpeed(0.0)
    time_step = int(driver.getBasicTimeStep())
    
    camera = driver.getDevice("camera_front")
    camera.enable(time_step)
    
    # 1. SOLUCIÓN VENTANA PEQUEÑA: Usamos WINDOW_NORMAL
    cv2.namedWindow("ESPIA (Haz clic en la linea)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ESPIA (Haz clic en la linea)", 800, 600) # Tamaño grande forzado

    print("--- MODO ESPÍA ACTIVO ---")
    print("1. Espera a que aparezca la ventana.")
    print("2. Haz CLIC IZQUIERDO sobre la línea amarilla/blanca.")
    print("3. Mira esta terminal para ver los números.")

    while driver.step() != -1:
        img_data = camera.getImage()
        if img_data:
            # Procesar imagen
            img = np.frombuffer(img_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            img = img[:, :, :3] # Quitar transparencia
            
            # Convertir a HSV para el análisis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Configurar el callback del mouse (le pasamos la imagen hsv)
            cv2.setMouseCallback("ESPIA (Haz clic en la linea)", pick_color, hsv)
            
            # Dibujar un círculo donde hicimos clic la última vez
            display_img = img.copy()
            if pixel_hsv is not None:
                # Texto en pantalla con los valores
                info = f"H:{pixel_hsv[0]} S:{pixel_hsv[1]} V:{pixel_hsv[2]}"
                cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("ESPIA (Haz clic en la linea)", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()