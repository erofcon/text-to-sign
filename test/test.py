import json
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from keras import Input, Model, Sequential
from keras.src.layers import Reshape, Flatten, LayerNormalization, MultiHeadAttention, RepeatVector
from matplotlib import pyplot as plt
from tensorflow.keras.layers import *
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


class SignLanguageDataProcessor:

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.mp_drawing = mp.solutions.drawing_utils
        self.FACE_DIM = 468 * 3  # x,y,z для 468 точек лица
        self.HAND_DIM = 21 * 3 * 2  # x,y,z для обеих рук
        self.POSE_DIM = 33 * 4  # x,y,z,visibility для позы
        self.TOTAL_DIM = self.FACE_DIM + self.HAND_DIM + self.POSE_DIM
        print(f"Всего признаков: {self.TOTAL_DIM}")

    def normalize_sequence(self, seq: np.ndarray, target: int = 30) -> np.ndarray:
        length, dim = seq.shape
        if length == target:
            return seq
        if length > target:
            idxs = np.linspace(0, length - 1, target)
            return np.vstack([
                np.interp(idxs, np.arange(length), seq[:, d])
                for d in range(dim)
            ]).T

        pad = target - length
        if length > 1:
            last = seq[-1]
            second_last = seq[-2]
            pads = np.array([
                second_last + (i + 1) / (pad + 1) * (last - second_last)
                for i in range(pad)
            ])
            return np.vstack([seq, pads])
        return np.vstack([seq, np.tile(seq[-1], (pad, 1))])

    def extract_frame_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        kp = np.zeros(self.TOTAL_DIM, dtype=float)
        idx = 0

        # Лицо
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
            )

            for lm in results.face_landmarks.landmark:
                kp[idx:idx + 3] = [lm.x, lm.y, lm.z]
                idx += 3
        else:
            idx += self.FACE_DIM
        # Левая рука
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
            )

            for lm in results.left_hand_landmarks.landmark:
                kp[idx:idx + 3] = [lm.x, lm.y, lm.z]
                idx += 3
        else:
            idx += self.HAND_DIM // 2
        # Правая рука
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )

            for lm in results.right_hand_landmarks.landmark:
                kp[idx:idx + 3] = [lm.x, lm.y, lm.z]
                idx += 3
        else:
            idx += self.HAND_DIM // 2
        # Поза
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )
            for lm in results.pose_landmarks.landmark:
                kp[idx:idx + 4] = [lm.x, lm.y, lm.z, lm.visibility]
                idx += 4
        return kp

    def extract_video_keypoints(self, path: str, max_frames: int = 50) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            print(f"Файл не найден: {path}")
            return None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)
        keypoints_list = []
        frame_idx = 0

        while len(keypoints_list) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                kp = self.extract_frame_keypoints(frame)
                if kp is not None:
                    keypoints_list.append(kp.tolist())

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) == ord('q'):
                break

            frame_idx += 1

        cap.release()
        if not keypoints_list:
            print("Ключевые точки не извлечены")
            return None
        return np.array(keypoints_list)

    def prepare_dataset_json(self, data_dir: str, output_json: str, max_frames: int = 50, target_seq: int = 30):
        dataset = []
        for word in os.listdir(data_dir):
            word_path = os.path.join(data_dir, word)

            if not os.path.isdir(word_path):
                continue
            for file in os.listdir(word_path):
                if not file.lower().endswith(('.mp4', '.avi', '.mov')):
                    continue

                video_path = os.path.join(word_path, file)
                kps = self.extract_video_keypoints(video_path, max_frames)

                if kps is None:
                    continue

                seq = self.normalize_sequence(kps, target_seq)
                dataset.append({
                    'word': word,
                    'sequence': seq.tolist()
                })

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Сохранено {len(dataset)} записей в {output_json}")


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Обнаружено GPU: {len(gpus)} устройство(а)")

    else:
        print("GPU не обнаружено, используется CPU")


class ImprovedSignLanguageModel:

    def __init__(self, vocab_size: int, sequence_length: int = 30, keypoint_dim: int = 1662):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.keypoint_dim = keypoint_dim
        self.model = None
        self.history = None

    def create_model(self, model_type: str = "transformer"):
        """
        Создание модели с выбором архитектуры

        Args:
            model_type: "simple", "lstm", "transformer"
        """
        if model_type == "simple":
            self.model = self._create_simple_model()
        elif model_type == "lstm":
            raise ValueError(f"Тип модели{model_type} не реализованно")
            # self.model = self._create_lstm_model()
        elif model_type == "transformer":
            # raise ValueError(f"Тип модели{model_type} не реализованно")
            self.model = self._create_transformer_model()
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        return self.model

    def _create_simple_model(self):
        """Простая fully-connected модель"""
        print("Создание простой модели...")

        # Вход
        word_input = Input(shape=(1,), name='word_input')

        # Эмбеддинг
        x = Embedding(self.vocab_size, 128, name='word_embedding')(word_input)
        x = Flatten()(x)

        # Полносвязные слои
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.sequence_length * self.keypoint_dim, activation='linear')(x)

        # Изменение формы для последовательности
        gesture_output = Reshape((self.sequence_length, self.keypoint_dim))(x)

        model = Model(inputs=word_input, outputs=gesture_output, name='SimpleSignModel')
        return model

    def _create_transformer_model(self):
        """Упрощенная Transformer модель"""
        print("Создание Transformer модели...")

        embed_dim = 256
        num_heads = 8
        ff_dim = 512

        # Вход
        word_input = Input(shape=(1,), name='word_input')

        # Эмбеддинг
        x = Embedding(self.vocab_size, embed_dim, name='word_embedding')(word_input)
        x = RepeatVector(self.sequence_length)(Flatten()(x))

        # Позиционное кодирование
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        pos_embedding = Embedding(self.sequence_length, embed_dim)(positions)
        x = x + pos_embedding

        # Transformer блок
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )(x, x)

        x = LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Feed forward
        ffn_output = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])(x)

        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Выходной слой
        gesture_output = Dense(self.keypoint_dim, activation='linear')(x)

        model = Model(inputs=word_input, outputs=gesture_output, name='TransformerSignModel')
        return model

    def load_model(self, filepath: str):
        """Загрузка модели"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Модель загружена: {filepath}")

    def compile_model(self, learning_rate: float = 0.001):
        """Компиляция модели"""
        if self.model is None:
            raise ValueError("Модель не создана")

        # Улучшенная функция потерь
        def combined_loss(y_true, y_pred):
            # Среднеквадратичная ошибка
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            # Плавность: разности между кадрами
            dt_true = y_true[:, 1:, :] - y_true[:, :-1, :]
            dt_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            smooth = tf.reduce_mean(tf.square(dt_true - dt_pred))
            return mse + 0.1 * smooth

            # Основная MSE потеря
            # mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            #
            # # Потеря плавности (smoothness loss)
            # diff_true = y_true[:, 1:, :] - y_true[:, :-1, :]
            # diff_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            # smoothness = tf.keras.losses.mean_squared_error(diff_true, diff_pred)
            #
            # return mse + 0.1 * smoothness

        # Компиляция
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=combined_loss,
            metrics=['mae', 'mse']
        )

        print("Модель скомпилирована")
        print(f"Параметры: {self.model.count_params():,}")

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 32):
        """Обучение модели"""
        if self.model is None:
            raise ValueError("Модель не создана и не скомпилирована")

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        print("Начинаем обучение...")

        # Обучение
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def save_model(self, filepath: str):
        """Сохранение модели"""
        if self.model is None:
            raise ValueError("Модель не создана")

        self.model.save(filepath)
        print(f"Модель сохранена: {filepath}")

    def load_model(self, filepath: str):
        """Загрузка модели"""

        def combined_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            dt_true = y_true[:, 1:, :] - y_true[:, :-1, :]
            dt_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            smooth = tf.reduce_mean(tf.square(dt_true - dt_pred))
            return mse + 0.1 * smooth

        # Загрузка с указанием custom_objects
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'combined_loss': combined_loss}
        )
        print(f"Модель загружена из: {filepath}")

    def predict_gesture(self, word_idx: int) -> np.ndarray:
        """Предсказание жеста для слова"""
        if self.model is None:
            raise ValueError("Модель не создана")

        input_data = np.array([[word_idx]])
        prediction = self.model.predict(input_data, verbose=0)

        return prediction[0]


def plot_training_history(history):
    """Улучшенная визуализация обучения"""
    if history is None:
        print("История обучения не найдена")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MSE
    axes[1, 0].plot(history.history['mse'], label='Training MSE', linewidth=2)
    axes[1, 0].plot(history.history['val_mse'], label='Validation MSE', linewidth=2)
    axes[1, 0].set_title('Mean Squared Error', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(history.history.get('lr', []), label='Learning Rate', linewidth=2)
    axes[1, 1].set_title('Learning Rate', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test, word_to_idx):
    """Оценка модели"""
    if model.model is None:
        raise ValueError("Модель не создана")

    print("Оценка модели...")

    # Предсказания
    y_pred = model.model.predict(X_test, verbose=0)

    # Общие метрики
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"Общий MSE: {mse:.4f}")
    print(f"Общий MAE: {mae:.4f}")

    # Метрики по словам
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    word_metrics = {}

    for i, word_idx in enumerate(X_test):
        word = idx_to_word[word_idx]
        if word not in word_metrics:
            word_metrics[word] = []

        word_mse = np.mean((y_test[i] - y_pred[i]) ** 2)
        word_metrics[word].append(word_mse)

    # Усреднение по словам
    for word in word_metrics:
        word_metrics[word] = np.mean(word_metrics[word])

    print("\nМетрики по словам:")
    for word, mse in sorted(word_metrics.items()):
        print(f"  {word}: MSE = {mse:.4f}")

    return {
        'mse': mse,
        'mae': mae,
        'word_metrics': word_metrics
    }


def visualize_gesture_prediction(gesture_sequence, title="Предсказанный жест"):
    """Визуализация предсказанного жеста"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Статистика по всем ключевым точкам
    axes[0, 0].plot(gesture_sequence.mean(axis=1), linewidth=2)
    axes[0, 0].set_title('Средние значения по времени')
    axes[0, 0].set_xlabel('Кадр')
    axes[0, 0].set_ylabel('Среднее значение')
    axes[0, 0].grid(True, alpha=0.3)

    # Стандартное отклонение
    axes[0, 1].plot(gesture_sequence.std(axis=1), linewidth=2, color='orange')
    axes[0, 1].set_title('Стандартное отклонение по времени')
    axes[0, 1].set_xlabel('Кадр')
    axes[0, 1].set_ylabel('Стандартное отклонение')
    axes[0, 1].grid(True, alpha=0.3)

    # Heatmap первых 100 признаков
    im = axes[1, 0].imshow(gesture_sequence[:, :100].T, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Heatmap ключевых точек (первые 100)')
    axes[1, 0].set_xlabel('Кадр')
    axes[1, 0].set_ylabel('Признак')
    plt.colorbar(im, ax=axes[1, 0])

    # Распределение значений
    axes[1, 1].hist(gesture_sequence.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 1].set_title('Распределение значений')
    axes[1, 1].set_xlabel('Значение')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def demo_prediction(model, word, word_to_idx, data_processor):
    """Демонстрация предсказания"""
    if word not in word_to_idx:
        print(f"Слово '{word}' не найдено в словаре")
        return

    word_idx = word_to_idx[word]

    # Предсказание
    predicted_gesture = model.predict_gesture(word_idx)

    # Денормализация
    if data_processor.is_fitted:
        original_shape = predicted_gesture.shape
        gesture_reshaped = predicted_gesture.reshape(-1, original_shape[-1])
        gesture_denormalized = data_processor.scaler.inverse_transform(gesture_reshaped)
        predicted_gesture = gesture_denormalized.reshape(original_shape)

    print(f"\nПредсказание для слова '{word}':")
    print(f"Форма: {predicted_gesture.shape}")
    print(f"Диапазон: [{predicted_gesture.min():.3f}, {predicted_gesture.max():.3f}]")
    print(f"Среднее: {predicted_gesture.mean():.3f}")
    print(f"Стандартное отклонение: {predicted_gesture.std():.3f}")

    # Визуализация
    visualize_gesture_prediction(predicted_gesture, f"Жест для слова '{word}'")

    return predicted_gesture


if __name__ == "__main__":

    # ------------------------
    # 1. Настройка и подготовка данных
    # ------------------------

    # Включаем GPU (если есть)

    data_directory = "data/videos"
    output_file = "data/dataset.json"

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    setup_gpu()

    processor = SignLanguageDataProcessor()
    if not os.path.isfile(output_file):
        print("Генерируем JSON‑датасет...")
        processor.prepare_dataset_json(
            data_dir=data_directory,
            output_json=output_file,
            max_frames=50,
            target_seq=30
        )

    print("Загружаем датасет из JSON...")
    with open(output_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    words = [entry["word"] for entry in raw]
    sequences = np.array([entry["sequence"] for entry in raw])

    le = LabelEncoder()
    X_indices = le.fit_transform(words)
    y = sequences

    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Сохранён LabelEncoder -> models/label_encoder.pkl")

    print(f"Всего примеров: {len(X_indices)}, классов: {len(le.classes_)}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_indices, y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ------------------------
    # 2. Построение и обучение модели
    # ------------------------

    # 2.a. Создаём модель (simple/lstm/transformer)
    model_wrapper = ImprovedSignLanguageModel(
        vocab_size=len(le.classes_),
        sequence_length=sequences.shape[1],
        keypoint_dim=sequences.shape[2]
    )

    model_wrapper.model = model_wrapper.create_model(model_type="transformer")

    model_wrapper.compile_model(learning_rate=1e-3)

    history = model_wrapper.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=16
    )

    plot_training_history(history)

    MODEL_PATH = os.path.join(MODEL_DIR, "transformer_weights.ckpt")
    model_wrapper.model.save_weights("models/transformer_weights.weights.h5")
    # model_wrapper.save_model(MODEL_PATH)

    new_wrapper = ImprovedSignLanguageModel(
        vocab_size=len(le.classes_),
        sequence_length=sequences.shape[1],
        keypoint_dim=sequences.shape[2]
    )

    new_wrapper.model = new_wrapper.create_model(model_type="transformer")
    # 2. Скомпилировать (чтобы инициализировать все слои)
    new_wrapper.compile_model(learning_rate=1e-3)
    # 3. Загрузить веса
    new_wrapper.model.load_weights("models/transformer_weights.weights.h5")

    # new_wrapper.load_model(MODEL_PATH)
    # new_wrapper.model.compile(  # после load нужно скомпилировать метрики
    #     optimizer="adam",
    #     loss=new_wrapper.model.loss,
    #     metrics=["mae", "mse"]
    # )

    print("Оценка на тестовой выборке:")
    results = new_wrapper.model.evaluate(
        x=np.expand_dims(X_test, axis=1),  # модель ожидает input shape=(None,1)
        y=y_test,
        verbose=1
    )
    print(f"Test loss, MAE, MSE: {results}")

    idx = X_test[0]
    pred_sequence = new_wrapper.predict_gesture(idx)
    word = le.inverse_transform([idx])[0]
    print(f"\nПример предсказания для слова «{word}»:")
    print(f"Форма последовательности: {pred_sequence.shape}")
    print(f"Диапазон значений: [{pred_sequence.min():.3f}, {pred_sequence.max():.3f}]")
