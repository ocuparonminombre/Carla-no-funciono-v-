import os
import csv
import torch
import torch.nn as nn

# 1. ENCONTRAR LA RUTA CORRECTA
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"--- CONFIGURANDO EN: {current_dir} ---")

# 2. CREAR EL CSV (HISTORIAL)
csv_path = os.path.join(current_dir, "historial_entrenamiento.csv")

try:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Escribimos los encabezados para las gráficas
        writer.writerow(["Episodio", "RecompensaTotal", "Pasos", "Epsilon"])
    print(f"✅ Historial creado exitosamente en:\n   -> {csv_path}")
except PermissionError:
    print(f"❌ ERROR: Cierra el archivo Excel {csv_path} antes de correr esto.")

# 3. CREAR UN CEREBRO VACÍO (.PTH)
# Esto evita errores de 'modelo no encontrado' en main_driver.py
model_path = os.path.join(current_dir, "modelo_entrenado.pth")

# Definimos la misma red neuronal que en dqn_agent.py para compatibilidad
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Entrada: 4 cuadros acumulados
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.layers = nn.Sequential(
            nn.Linear(2304, 512), nn.ReLU(),  # <--- CAMBIAR 3136 POR 2304
            nn.Linear(512, 3)
        )
    def forward(self, x):
        return self.layers(self.features(x))

# Solo creamos el cerebro si no existe uno (para no borrar uno bueno por accidente)
# Si quieres forzar el borrado, elimina el archivo .pth manualmente antes de correr esto.
if not os.path.exists(model_path):
    dummy_model = DQN()
    torch.save(dummy_model.state_dict(), model_path)
    print(f"✅ Cerebro inicial creado en:\n   -> {model_path}")
else:
    print(f"ℹ️ El cerebro ya existe, no lo sobreescribimos por seguridad.")

print("\n--- ¡LISTO! AHORA EJECUTA MAIN_DRIVER.PY ---")
input("Presiona Enter para salir...")