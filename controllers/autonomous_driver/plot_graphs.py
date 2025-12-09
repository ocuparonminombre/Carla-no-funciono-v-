import pandas as pd
import matplotlib.pyplot as plt
import os

# Ruta al archivo CSV
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "historial_entrenamiento.csv")

def plotear():
    if not os.path.exists(csv_path):
        print("❌ No encuentro el historial. Entrena un poco primero.")
        return

    try:
        # Cargar datos
        df = pd.read_csv(csv_path)
        
        if len(df) < 2:
            print("⚠️ Muy pocos datos para graficar. Espera unos episodios más.")
            return

        # Calcular Media Móvil (Suavizado) para ver la tendencia clara
        # Window=50 significa que promedia los últimos 50 intentos
        df['Media_Movil'] = df['RecompensaTotal'].rolling(window=20).mean()

        plt.figure(figsize=(12, 6))

        # --- GRÁFICA 1: APRENDIZAJE ---
        plt.subplot(1, 2, 1)
        plt.plot(df['Episodio'], df['RecompensaTotal'], color='lightgray', alpha=0.5, label='Ruido (Por episodio)')
        plt.plot(df['Episodio'], df['Media_Movil'], color='blue', linewidth=2, label='Tendencia (Aprendizaje)')
        plt.title('Curva de Aprendizaje')
        plt.xlabel('Episodios')
        plt.ylabel('Puntos (Recompensa)')
        plt.legend()
        plt.grid(True)

        # --- GRÁFICA 2: EXPLORACIÓN ---
        plt.subplot(1, 2, 2)
        plt.plot(df['Episodio'], df['Epsilon'], color='orange', linewidth=2)
        plt.title('Nivel de Exploración (Epsilon)')
        plt.xlabel('Episodios')
        plt.ylabel('Probabilidad de hacer locuras')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        print("✅ Gráfica generada.")

    except Exception as e:
        print(f"Error al graficar: {e}")
        print("Asegúrate de CERRAR el archivo CSV en Excel antes de correr esto.")

if __name__ == "__main__":
    plotear()