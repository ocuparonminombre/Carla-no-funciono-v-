import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
<<<<<<< HEAD
        # --- CONFIGURACIÓN DE COLORES ---
        
        # AMARILLO (Líneas centrales)
        self.lower_yellow = np.array([15, 89, 124])
        self.upper_yellow = np.array([35, 255, 255])
        
        # BLANCO (Líneas laterales)
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        
        # NARANJA (Conos - Rango Estricto para no confundir con amarillo)
        self.lower_orange = np.array([5, 150, 150])
        self.upper_orange = np.array([15, 255, 255])
=======
        # COLORES
        self.lower_yellow = np.array([15, 89, 124])
        self.upper_yellow = np.array([35, 255, 255])
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        
        # --- MEMORIA (NUEVO) ---
        # Guardamos el último error conocido.
        # Si perdemos la línea, usamos este valor.
        self.last_error = 0.0
>>>>>>> 0b03da00f280a85e9d71e9af7573a02793c55387

    def process_image(self, image):
        if image is None: return image, 0

        height, width, _ = image.shape
        
        # 1. RECORTAR (ROI)
<<<<<<< HEAD
        start_y = int(height * 0.45)
        end_y = int(height * 0.90)
        roi = image[start_y:end_y, :].copy()
        roi_h, roi_w, _ = roi.shape

        # 2. MÁSCARAS DE COLOR
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        mask_orange = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        
        # Combinamos todo en una máscara bruta
        mask_raw = cv2.bitwise_or(mask_yellow, mask_white)
        mask_raw = cv2.bitwise_or(mask_raw, mask_orange)

        # 3. FILTRO ANTI-RUIDO (Limpiar máscara)
        # Creamos una máscara negra limpia
        mask_clean = np.zeros_like(mask_raw)
        
        contours, _ = cv2.findContours(mask_raw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # FILTRO: Solo aceptamos objetos grandes (> 200px)
            # Esto elimina los triángulos pequeños del suelo ("dientes de tiburón")
            if area > 200:
                # Dibujamos en la máscara para la IA (Relleno blanco)
                cv2.drawContours(mask_clean, [cnt], -1, 255, -1)
                # Dibujamos en la pantalla visual (Borde VERDE)
                cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            else:
                # Lo que ignoramos (Ruido), lo pintamos de ROJO para que tú sepas
                cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 1)

        # 4. NAVEGACIÓN (Usando mask_clean)
        M = cv2.moments(mask_clean)
        image_center = int(roi_w / 2)
        cx = image_center
        detected = False
        
        if M["m00"] > 100:
            cx = int(M["m10"] / M["m00"])
            detected = True
            cv2.line(roi, (cx, 0), (cx, roi_h), (0, 255, 0), 4)

        cv2.line(roi, (image_center, 0), (image_center, roi_h), (255, 0, 0), 2)
        
        error = (cx - image_center) / (roi_w / 2)
        
        if not detected: 
            error = 0 
=======
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
>>>>>>> 0b03da00f280a85e9d71e9af7573a02793c55387

        return roi, error