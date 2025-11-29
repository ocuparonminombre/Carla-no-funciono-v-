import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        # COLORES
        self.lower_yellow = np.array([15, 89, 124])
        self.upper_yellow = np.array([35, 255, 255])
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        
        # --- MEMORIA (NUEVO) ---
        # Guardamos el último error conocido.
        # Si perdemos la línea, usamos este valor.
        self.last_error = 0.0

    def process_image(self, image):
        if image is None: return image, 0

        height, width, _ = image.shape
        
        # 1. RECORTAR (ROI)
        # Cortamos menos (0.5) para aprovechar el nuevo FOV ancho
        cut_height = int(height * 0.55) 
        roi = image[cut_height:, :].copy() 
        roi_h, roi_w, _ = roi.shape

        # 2. DETECCIÓN
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        mask_lanes = cv2.bitwise_or(mask_yellow, mask_white)

        # 3. DIBUJAR
        contours, _ = cv2.findContours(mask_lanes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)

        # 4. NAVEGACIÓN CON MEMORIA
        M = cv2.moments(mask_lanes)
        cx = int(roi_w / 2)
        detected = False
        
        # Umbral más bajo (50) porque al estar lejos se ven más chicas las líneas
        if M["m00"] > 50:
            cx = int(M["m10"] / M["m00"])
            detected = True
            
            # Línea VERDE (Objetivo actual)
            cv2.line(roi, (cx, 0), (cx, roi_h), (0, 255, 0), 4)
            
            # Calculamos el error actual
            car_center = int(roi_w / 2)
            error = (cx - car_center) / (roi_w / 2)
            
            # ACTUALIZAMOS LA MEMORIA
            self.last_error = error
            
        else:
            # --- AQUÍ ESTÁ LA MAGIA ---
            # No detectamos nada. ¿Qué hacemos?
            # En lugar de decir "error = 0" (que confunde a la IA),
            # le recordamos el último error pero un poco más exagerado.
            
            # Si la ultima vez estaba a la izquierda, asumimos que sigue yéndose a la izquierda
            error = self.last_error * 1.2 
            
            # Limitamos para que no se vuelva loco (-1 a 1)
            error = max(min(error, 1.0), -1.0)
            
            # Indicador visual de "PERDIDO" (Texto Rojo)
            cv2.putText(roi, "LINEA PERDIDA - USANDO MEMORIA", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Línea AZUL (Auto)
        car_center = int(roi_w / 2)
        cv2.line(roi, (car_center, 0), (car_center, roi_h), (255, 0, 0), 2)

        return roi, error