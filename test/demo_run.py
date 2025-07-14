# demo_run_fixed.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
from test import SignLanguageDataProcessor, ImprovedSignLanguageModel, setup_gpu

# ------------------------
# 1. Подготовка
# ------------------------
setup_gpu()

# Пути
ENCODER_PATH = "models/label_encoder.pkl"
WEIGHTS_PATH = "models/transformer_weights.weights.h5"

# Загружаем LabelEncoder
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Параметры модели (эти же, что были при обучении)
vocab_size   = len(le.classes_)
seq_len      = 30                # или тот, что вы использовали
keypoint_dim = SignLanguageDataProcessor().TOTAL_DIM

# ------------------------
# 2. Воссоздаём модель и загружаем веса
# ------------------------
# Создаём обёртку
wrapper = ImprovedSignLanguageModel(
    vocab_size=vocab_size,
    sequence_length=seq_len,
    keypoint_dim=keypoint_dim
)
# Шаг 1: создаём архитектуру transformer
wrapper.model = wrapper.create_model(model_type="transformer")
# Шаг 2: (опционально) компилируем — нужно, чтобы инициализировать optimizer state
wrapper.compile_model(learning_rate=1e-3)
# Шаг 3: загружаем веса
wrapper.model.load_weights(WEIGHTS_PATH)  
# — предупреждение про optimizer можно игнорировать для инференса

# ------------------------
# 3. Предсказание и визуализация (2D pose)
# ------------------------
def visualize_pose_sequence(sequence: np.ndarray, word: str):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.animation import FuncAnimation

    # схема соединений из MediaPipe Pose
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
        (9,10),
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(25,27),
        (24,26),(26,28),(27,31),(28,32),
        (15,17),(16,18),(17,19),(18,20),(11,22),(12,21)
    ]

    def extract_pose(frame: np.ndarray):
        start = 468*3 + 21*3*2
        pose = frame[start:].reshape(33,4)[:, :3]
        return pose

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.set_title(f"Жест: {word}")

    scat = ax.scatter([], [], [], c='red')
    lines = [ax.plot([],[],[],c='blue')[0] for _ in POSE_CONNECTIONS]

    def init():
        scat._offsets3d = ([],[],[])
        for ln in lines:
            ln.set_data([],[]); ln.set_3d_properties([])
        return [scat, *lines]

    def update(i):
        pose3d = extract_pose(sequence[i])
        x,y,z = pose3d[:,0], pose3d[:,1], pose3d[:,2]
        scat._offsets3d = (x,y,z)
        for idx,(a,b) in enumerate(POSE_CONNECTIONS):
            xa,ya,za = pose3d[a]
            xb,yb,zb = pose3d[b]
            lines[idx].set_data([xa,xb],[ya,yb])
            lines[idx].set_3d_properties([za,zb])
        return [scat, *lines]

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=sequence.shape[0], interval=100, blit=True)
    plt.show()


# Вводим слово и прогоняем инференс
if __name__ == "__main__":
    while True:
        word = input("Введите слово (или 'exit'): ").strip()
        if word.lower() in ("exit","quit"):
            break
        if word not in le.classes_:
            print("Слово не в словаре:", le.classes_)
            continue

        idx = le.transform([word])[0]
        seq = wrapper.predict_gesture(idx)  # (30, 1662)
        visualize_pose_sequence(seq, word)
