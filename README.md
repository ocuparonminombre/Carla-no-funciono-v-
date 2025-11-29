# Carla-no-funciono-v-
Plan B Carla funciono pero nuestras pcs no logran correrlo asi que F

nombre-repo-conduccion-autonoma/
│
├── controllers/                  # AQUÍ ESTÁ EL CEREBRO (Python)
│   └── autonomous_driver/        # Controlador principal del vehículo
│       ├── main_driver.py        # Script principal que ejecuta Webots
│       ├── vision/               # Módulo de Jhessica
│       │   ├── __init__.py
│       │   ├── lane_detector.py  # Detección de líneas (OpenCV)
│       │   ├── traffic_light.py  # Detección de semáforos
│       │   └── yolo_utils.py     # Detección de objetos
│       ├── rl/                   # Módulo de Nicolas
│       │   ├── __init__.py
│       │   ├── dqn_agent.py      # Agente DQN (Red Neuronal)
│       │   ├── rewards.py        # Sistema de recompensas (+10, -100, etc.)
│       │   └── models/           # Donde se guardan los .pth o .h5 entrenados
│       └── utils/
│           └── preprocessing.py  # Limpieza de datos antes de entrar a la red
│
├── worlds/                       # TRABAJO DE CARLOS (Entorno)
│   ├── city_training.wbt         # Mapa de entrenamiento
│   ├── city_testing.wbt          # Mapa de pruebas final
│   └── textures/                 # Texturas personalizadas si usan
│
├── protos/                       # Objetos personalizados (si crean señales nuevas)
│
├── docs/                         # Documentación del proyecto
│   ├── arquitectura.md
│   └── manual_instalacion.md
│
├── requirements.txt              # Lista de librerías (torch, opencv-python, etc.)
├── .gitignore                    # ¡Muy importante!
└── README.md                     # Presentación del proyecto

agrengamos aprendizaje por curriculo

python .\controllers\autonomus_driver\main_driver.py


instalar webots
usar pip install -r requirements.txt



para iniciar las pruebas python controllers/autonomous_driver/main_driver.py
python -c "import torch; print(torch.__version__)"


