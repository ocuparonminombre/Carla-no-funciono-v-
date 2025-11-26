import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        # --- CALIBRACIÓN FINAL (Tus valores detectados) ---
        self.lower_yellow = np.array([15, 89, 124])
        self.upper_yellow = np.array([35, 255, 255])
        
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])

    def process_image(self, image):
        if image is None:
            return image, 0

        # 1. Recortar (ROI) y hacer COPIA para poder dibujar
        height, width, _ = image.shape
        # --- AQUÍ ESTÁ EL FIX: .copy() ---
        # Esto crea una imagen nueva editable en lugar de usar la memoria de solo lectura
        roi = image[int(height/2):, :].copy()
        
        # 2. Convertir a HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 3. Filtrar colores
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        mask_combined = cv2.bitwise_or(mask_yellow, mask_white)
        
        # 4. Cálculo del Centro
        M = cv2.moments(mask_combined)
        cx = int(width / 2)
        
        detected = False
        if M["m00"] > 100:
            cx = int(M["m10"] / M["m00"])
            detected = True
            
            # Dibujos (Ahora sí funcionarán porque roi es una copia)
            cv2.line(roi, (cx, 0), (cx, height), (0, 255, 0), 5)
            cv2.circle(roi, (cx, int(height/2)), 10, (0, 255, 0), -1)
            
            contours, _ = cv2.findContours(mask_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)
            
        car_center = int(width / 2)
        cv2.line(roi, (car_center, 0), (car_center, height), (255, 0, 0), 2)
        
        # 5. Calcular Error
        error = (cx - car_center) / (width / 2)
        
        if not detected:
            error = 0

        return roi, error