import json
import re
import sqlite3
import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pymorphy3 import MorphAnalyzer

# Импорт функций экспорта
# from export import export_sentence_for_avatar, export_single_sign_for_avatar
#
@dataclass
class SignData:
    word: str
    base: str
    fps: float
    keypoints: np.ndarray  # (frames, 75, 4)
    synonyms: List[str]

    def __post_init__(self):
        if self.keypoints.ndim != 3 or self.keypoints.shape[1:] != (75, 4):
            raise ValueError(f"Ожидается ключевых точек (N,75,4), получено {self.keypoints.shape}")



def export_sentence_for_avatar(sentence_data: SignData, output_file="animation_data.json"):

    animation_data = {
        "text": sentence_data.text,
        "fps": sentence_data.fps,
        "word_boundaries": sentence_data.word_boundaries,
        "total_frames": int(len(sentence_data.merged_keypoints)),
        "frames": []
    }
    # Обработка каждого кадра
    for frame_keypoints in sentence_data.merged_keypoints:
        frame_data = process_keypoints_to_avatar_frame(frame_keypoints)
        animation_data["frames"].append(frame_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(animation_data, f, ensure_ascii=False, indent=2)
    print(f"Данные анимации экспортированы в {output_file}")
    print(f"Всего кадров: {len(animation_data['frames'])}")
    print(f"Длительность: {len(animation_data['frames']) / sentence_data.fps:.2f} сек.")
    return output_file


def process_keypoints_to_avatar_frame(keypoints):
    """
    Преобразовать ключевые точки из MediaPipe в формат кадра для анимации 3D аватара.
    Используются следующие ключевые точки:
    11: LEFT_SHOULDER, 12: RIGHT_SHOULDER
    13: LEFT_ELBOW, 14: RIGHT_ELBOW
    15: LEFT_WRIST, 16: RIGHT_WRIST
    0: NOSE (для головы)
    """
    frame = {
        "leftArm": {"rotation": {"x": 0, "y": 0, "z": 0}},
        "rightArm": {"rotation": {"x": 0, "y": 0, "z": 0}},
        "head": {"rotation": {"x": 0, "y": 0, "z": 0}},
        "leftHand": {"position": {"x": 0, "y": 0, "z": 0}},
        "rightHand": {"position": {"x": 0, "y": 0, "z": 0}}
    }
    if len(keypoints) >= 33:
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_elbow = keypoints[13]
        right_elbow = keypoints[14]
        left_wrist = keypoints[15]
        right_wrist = keypoints[16]
        nose = keypoints[0]
        if left_shoulder[3] > 0.5 and left_wrist[3] > 0.5:
            dx = left_wrist[0] - left_shoulder[0]
            dy = left_wrist[1] - left_shoulder[1]
            dz = left_wrist[2] - left_shoulder[2]
            frame["leftArm"]["rotation"]["z"] = float(np.arctan2(dy, dx))
            frame["leftArm"]["rotation"]["y"] = float(np.arctan2(dz, np.sqrt(dx * dx + dy * dy)))
            frame["leftHand"]["position"]["x"] = float(left_wrist[0])
            frame["leftHand"]["position"]["y"] = float(-left_wrist[1])
            frame["leftHand"]["position"]["z"] = float(left_wrist[2])
        if right_shoulder[3] > 0.5 and right_wrist[3] > 0.5:
            dx = right_wrist[0] - right_shoulder[0]
            dy = right_wrist[1] - right_shoulder[1]
            dz = right_wrist[2] - right_shoulder[2]
            frame["rightArm"]["rotation"]["z"] = float(np.arctan2(dy, dx))
            frame["rightArm"]["rotation"]["y"] = float(np.arctan2(dz, np.sqrt(dx * dx + dy * dy)))
            frame["rightHand"]["position"]["x"] = float(right_wrist[0])
            frame["rightHand"]["position"]["y"] = float(-right_wrist[1])
            frame["rightHand"]["position"]["z"] = float(right_wrist[2])
        if nose[3] > 0.5:
            frame["head"]["rotation"]["y"] = float((nose[0] - 0.5) * 0.5)
            frame["head"]["rotation"]["x"] = float((0.5 - nose[1]) * 0.3)
    return frame


def export_single_sign_for_avatar(sign_data, output_file=None):
    """
    Экспортировать данные одного знака в формате JSON для анимации 3D аватара
    """
    if output_file is None:
        output_file = f"{sign_data.word}_animation.json"
    animation_data = {
        "text": sign_data.word,
        "fps": sign_data.fps,
        "word_boundaries": [[0, int(len(sign_data.keypoints) - 1)]],
        "total_frames": int(len(sign_data.keypoints)),
        "frames": []
    }
    for frame_keypoints in sign_data.keypoints:
        frame_data = process_keypoints_to_avatar_frame(frame_keypoints)
        animation_data["frames"].append(frame_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(animation_data, f, ensure_ascii=False, indent=2)
    print(f"Данные анимации для знака '{sign_data.word}' экспортированы в {output_file}")
    return output_file



@dataclass
class SentenceData:

    text: str
    signs: List[SignData]
    merged_keypoints: np.ndarray  # Объединённые ключевые точки для всех знаков
    fps: float
    word_boundaries: List[Tuple[int, int]]  # (начальный кадр, конечный кадр) для каждого знака
    pause_frames: int = 10  # Кадров паузы между словами

    def get_word_at_frame(self, frame_idx: int) -> Optional[str]:
        """Получить слово, которое подписывается в указанном кадре"""
        for i, (start, end) in enumerate(self.word_boundaries):
            if start <= frame_idx <= end:
                return self.signs[i].word
        return None



class SignDB:
    def __init__(self, db_path: str = 'signs.db', data_dir: str = 'sign_data'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def _init_tables(self):

        queries = [
            '''CREATE TABLE IF NOT EXISTS signs (
                word TEXT PRIMARY KEY,
                base TEXT,
                fps REAL,
                file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''',
            '''CREATE TABLE IF NOT EXISTS synonyms (
                word TEXT,
                synonym TEXT,
                PRIMARY KEY(word, synonym)
            )''',
            '''CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                words TEXT,
                file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )'''
        ]
        c = self.conn.cursor()
        for query in queries:
            c.execute(query)
        self.conn.commit()

    def add(self, sign: SignData, filepath: str):
        """Добавить или обновить знак в базе данных"""
        c = self.conn.cursor()
        c.execute('REPLACE INTO signs (word, base, fps, file) VALUES (?, ?, ?, ?)',
                  (sign.word, sign.base, sign.fps, filepath))
        c.execute('DELETE FROM synonyms WHERE word = ?', (sign.word,))
        for syn in sign.synonyms:
            c.execute('INSERT INTO synonyms (word, synonym) VALUES (?, ?)', (sign.word, syn))
        self.conn.commit()

    def get(self, token: str) -> Optional[SignData]:
        """Получить данные знака по слову, синониму или базовой форме"""
        c = self.conn.cursor()
        queries = [
            'SELECT word, base, fps, file FROM signs WHERE word=?',
            '''SELECT s.word, s.base, s.fps, s.file 
               FROM signs s JOIN synonyms y ON s.word=y.word WHERE y.synonym=?''',
            'SELECT word, base, fps, file FROM signs WHERE base=?'
        ]

        row = None
        for query in queries:
            row = c.execute(query, (token,)).fetchone()
            if row:
                break
        if not row:
            return None

        word, base, fps, file = row
        try:
            data = np.load(file)
            kp = data['keypoints']
            syns = [r[0] for r in c.execute('SELECT synonym FROM synonyms WHERE word=?', (word,))]
            return SignData(word, base, fps, kp, syns)
        except Exception as e:
            print(f"Ошибка загрузки {word}: {e}")
            return None

    def save_sentence(self, sentence: SentenceData, filepath: str):

        c = self.conn.cursor()
        words_str = ','.join([sign.word for sign in sentence.signs])
        c.execute('INSERT INTO sentences (text, words, file) VALUES (?, ?, ?)',
                  (sentence.text, words_str, filepath))
        self.conn.commit()

    def list_all(self) -> List[str]:

        c = self.conn.cursor()
        return [r[0] for r in c.execute('SELECT word FROM signs ORDER BY word')]

    def list_sentences(self) -> List[Tuple[int, str]]:

        c = self.conn.cursor()
        return [(r[0], r[1]) for r in c.execute('SELECT id, text FROM sentences ORDER BY created_at DESC')]

    def close(self):
        self.conn.close()


# ==================== Построитель предложений ====================
class SentenceBuilder:
    def __init__(self, pause_frames: int = 10, transition_frames: int = 5):
        self.pause_frames = pause_frames
        self.transition_frames = transition_frames

    def create_neutral_pose(self, reference_frame: np.ndarray) -> np.ndarray:

        neutral = reference_frame.copy()
        # Левая рука (индексы 33-53)
        neutral[33:54, :3] = 0
        neutral[33:54, 3] = 0.3
        # Правая рука (индексы 54-74)
        neutral[54:75, :3] = 0
        neutral[54:75, 3] = 0.3
        return neutral

    def interpolate_keypoints(self, start_frame: np.ndarray, end_frame: np.ndarray,
                              num_frames: int) -> np.ndarray:

        if num_frames <= 0:
            return np.array([])
        interpolated = np.zeros((num_frames, 75, 4))
        for i in range(num_frames):
            alpha = i / (num_frames - 1) if num_frames > 1 else 0
            interpolated[i, :, :3] = (1 - alpha) * start_frame[:, :3] + alpha * end_frame[:, :3]
            interpolated[i, :, 3] = np.minimum(start_frame[:, 3], end_frame[:, 3])
        return interpolated

    def merge_signs(self, signs: List[SignData], target_fps: float = 30.0) -> SentenceData:

        if not signs:
            raise ValueError("Нет доступных знаков для объединения")
        normalized_signs = []
        for sign in signs:
            if sign.fps != target_fps:
                original_frames = len(sign.keypoints)
                target_frames = int(original_frames * target_fps / sign.fps)
                indices = np.linspace(0, original_frames - 1, target_frames)
                resampled_keypoints = np.zeros((target_frames, 75, 4))
                for i, idx in enumerate(indices):
                    if idx == int(idx):
                        resampled_keypoints[i] = sign.keypoints[int(idx)]
                    else:
                        lower_idx = int(idx)
                        upper_idx = min(lower_idx + 1, original_frames - 1)
                        alpha = idx - lower_idx
                        resampled_keypoints[i] = (1 - alpha) * sign.keypoints[lower_idx] + alpha * sign.keypoints[
                            upper_idx]
                normalized_sign = SignData(sign.word, sign.base, target_fps, resampled_keypoints, sign.synonyms)
                normalized_signs.append(normalized_sign)
            else:
                normalized_signs.append(sign)

        merged_keypoints = []
        word_boundaries = []
        current_frame = 0
        for i, sign in enumerate(normalized_signs):
            if i > 0:
                prev_sign = normalized_signs[i - 1]
                transition = self.interpolate_keypoints(prev_sign.keypoints[-1], sign.keypoints[0],
                                                        self.transition_frames)
                if len(transition) > 0:
                    merged_keypoints.extend(transition)
                    current_frame += self.transition_frames
            start_frame = current_frame
            merged_keypoints.extend(sign.keypoints)
            current_frame += len(sign.keypoints)
            end_frame = current_frame - 1
            word_boundaries.append((start_frame, end_frame))
            if i < len(normalized_signs) - 1:
                neutral_pose = self.create_neutral_pose(sign.keypoints[-1])
                pause_keypoints = np.tile(neutral_pose, (self.pause_frames, 1, 1))
                merged_keypoints.extend(pause_keypoints)
                current_frame += self.pause_frames
        merged_array = np.array(merged_keypoints)
        sentence_text = ' '.join([sign.word for sign in signs])
        return SentenceData(
            text=sentence_text,
            signs=normalized_signs,
            merged_keypoints=merged_array,
            fps=target_fps,
            word_boundaries=word_boundaries,
            pause_frames=self.pause_frames
        )


class Processor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.morph = MorphAnalyzer()
        self.db = SignDB()
        self.sentence_builder = SentenceBuilder()

    def extract_keypoints(self, video_path: str) -> Tuple[np.ndarray, float]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.holistic.process(rgb)
            frames.append(self._extract_frame_keypoints(res))
        cap.release()
        if not frames:
            raise ValueError("Кадры не извлечены")
        arr = np.array(frames, dtype=np.float32)
        hip_l = mp.solutions.pose.PoseLandmark.LEFT_HIP.value
        hip_r = mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
        for i in range(arr.shape[0]):
            root = arr[i, [hip_l, hip_r], :3].mean(axis=0)
            arr[i, :, :3] -= root
        return arr, fps

    def _extract_frame_keypoints(self, res) -> np.ndarray:

        pts = []
        # Поза (33 точки)
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark:
                pts.append([lm.x, lm.y, lm.z, lm.visibility])
        else:
            pts.extend([[0, 0, 0, 0]] * 33)
        # Левая рука (21 точка)
        if res.left_hand_landmarks:
            for lm in res.left_hand_landmarks.landmark:
                pts.append([lm.x, lm.y, lm.z, 1])
        else:
            pts.extend([[0, 0, 0, 0]] * 21)
        # Правая рука (21 точка)
        if res.right_hand_landmarks:
            for lm in res.right_hand_landmarks.landmark:
                pts.append([lm.x, lm.y, lm.z, 1])
        else:
            pts.extend([[0, 0, 0, 0]] * 21)
        return np.array(pts, dtype=np.float32)

    def add_sign(self, word: str, video: str, synonyms: Optional[List[str]] = None):

        syns = synonyms or []
        base = self.morph.parse(word)[0].normal_form
        kp, fps = self.extract_keypoints(video)
        filepath = self.db.data_dir / f"{base}.npz"
        np.savez_compressed(filepath, keypoints=kp)
        self.db.add(SignData(word, base, fps, kp, syns), str(filepath))
        print(f"Знак '{word}' (базовая форма: '{base}') добавлен с {len(kp)} кадрами.")

    def translate_sentence(self, text: str, save_merged: bool = False) -> SentenceData:

        tokens = re.findall(r"\w+", text.lower())
        signs = []
        missing_words = []
        print(f"Перевод предложения: '{text}'")
        print(f"Найденные токены: {tokens}")
        for token in tokens:
            base = self.morph.parse(token)[0].normal_form
            sign = self.db.get(token) or self.db.get(base)
            if sign:
                signs.append(sign)
                print(f"✓ Найден знак для '{token}': {sign.word}")
            else:
                missing_words.append(token)
                print(f"✗ Не найден знак для '{token}'")
        if missing_words:
            print(f"\nВнимание: отсутствуют знаки для: {', '.join(missing_words)}")
            print("Предложение будет неполным.")
        if not signs:
            raise ValueError("Не найдено ни одного знака для перевода")
        sentence = self.sentence_builder.merge_signs(signs)
        if save_merged:
            filename = self.db.data_dir / f"sentence_{int(time.time())}.npz"
            np.savez_compressed(filename,
                                keypoints=sentence.merged_keypoints,
                                word_boundaries=sentence.word_boundaries,
                                words=[sign.word for sign in signs])
            self.db.save_sentence(sentence, str(filename))
            print(f"Объединённое предложение сохранено в {filename}")
        return sentence

    def translate(self, text: str) -> List[SignData]:

        tokens = re.findall(r"\w+", text.lower())
        result = []
        for t in tokens:
            base = self.morph.parse(t)[0].normal_form
            s = self.db.get(t) or self.db.get(base)
            if s:
                result.append(s)
                print(f"✓ Соответствие для '{t}': {s.word}")
            else:
                print(f"✗ Не найден знак для '{t}'")
        return result

    # Методы экспорта для 3D аватара
    def export_sentence_animation(self, text: str, output_file: str = None):

        sentence = self.translate_sentence(text)
        if output_file is None:
            output_file = f"sentence_{text.replace(' ', '_')}.json"
        return export_sentence_for_avatar(sentence, output_file)

    def export_word_animation(self, word: str, output_file: str = None):

        sign = self.db.get(word)
        if not sign:
            raise ValueError(f"Знак для '{word}' не найден")
        return export_single_sign_for_avatar(sign, output_file)


# ==================== Визуализатор ====================
class Visualizer:
    def __init__(self, elev=30, azim=-60, roll=0):
        self.pose_conns = np.array(list(mp.solutions.pose.POSE_CONNECTIONS))
        self.hand_conns = np.array(list(mp.solutions.hands.HAND_CONNECTIONS))
        self.elev = elev
        self.azim = azim
        self.roll = roll
        self.view_presets = {
            'front': {'elev': 0, 'azim': 0},
            'side': {'elev': 0, 'azim': -90},
            'top': {'elev': 90, 'azim': 0},
            'bottom': {'elev': -90, 'azim': 0},
            'isometric': {'elev': 30, 'azim': -60},
            'profile': {'elev': 0, 'azim': 90},
            'back': {'elev': 0, 'azim': 180},
            'diagonal': {'elev': 45, 'azim': -45}
        }

    def set_view_preset(self, preset_name: str):
        """Установить предустановленный режим просмотра"""
        if preset_name in self.view_presets:
            preset = self.view_presets[preset_name]
            self.elev = preset['elev']
            self.azim = preset['azim']
            print(f"Режим просмотра установлен как {preset_name}: elevation={self.elev}, azimuth={self.azim}")
        else:
            print(f"Неизвестный режим '{preset_name}'. Доступно: {list(self.view_presets.keys())}")

    def _setup_3d_axis(self, ax, title: str):
        """Настройка 3D области для визуализации"""
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=12, pad=20)
        ax.view_init(elev=self.elev, azim=self.azim, roll=self.roll)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')

    def _draw_connections(self, ax, pts, conns, offset, count, color, alpha=0.7):

        for i, j in conns:
            if (offset <= i < offset + count and offset <= j < offset + count and
                    pts[i, 3] > 0.5 and pts[j, 3] > 0.5):
                ax.plot3D([pts[i, 0], pts[j, 0]],
                          [pts[i, 1], pts[j, 1]],
                          [pts[i, 2], pts[j, 2]],
                          color=color, alpha=alpha, linewidth=2)

    def _draw_points(self, ax, pts, offset, count, color, size=30):

        visible_mask = pts[offset:offset + count, 3] > 0.5
        if not visible_mask.any():
            return
        visible_pts = pts[offset:offset + count][visible_mask]
        ax.scatter(visible_pts[:, 0], visible_pts[:, 1], visible_pts[:, 2],
                   c=color, s=size, alpha=0.8, edgecolors='black', linewidth=0.5)

    def animate_sentence(self, sentence: SentenceData, save: Optional[str] = None):

        plt.style.use('default')
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111, projection='3d')

        progress_text = fig.text(0.02, 0.95, '', fontsize=10,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        current_word_text = fig.text(0.02, 0.88, '', fontsize=12, weight='bold',
                                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
        sentence_text = fig.text(0.02, 0.81, f"Предложение: {sentence.text}", fontsize=10,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        total_frames = len(sentence.merged_keypoints)
        interval = max(50, min(200, int(1000 / sentence.fps)))
        current_elev = self.elev
        current_azim = self.azim
        current_roll = self.roll

        def init():
            nonlocal current_elev, current_azim, current_roll
            self._setup_3d_axis(ax, f"Перевод: {sentence.text}")
            progress_text.set_text(f"Кадр: 1/{total_frames} | FPS: {sentence.fps:.1f}")
            current_word_text.set_text("Текущее слово: (начало)")
            current_elev = ax.elev
            current_azim = ax.azim
            return []

        def update(frame_idx):
            nonlocal current_elev, current_azim, current_roll
            current_elev = ax.elev
            current_azim = ax.azim
            current_word = sentence.get_word_at_frame(frame_idx)
            word_display = current_word if current_word else "(переход/пауза)"
            progress_text.set_text(f"Кадр: {frame_idx + 1}/{total_frames} | FPS: {sentence.fps:.1f} | " +
                                   f"Прогресс: {(frame_idx + 1) / total_frames * 100:.1f}%")
            current_word_text.set_text(f"Текущее слово: {word_display}")
            ax.clear()
            self._setup_3d_axis(ax, f"Перевод: {sentence.text}")
            ax.view_init(elev=current_elev, azim=current_azim, roll=current_roll)
            frame = sentence.merged_keypoints[frame_idx]
            self._draw_connections(ax, frame, self.pose_conns, 0, 33, 'blue', 0.8)
            self._draw_points(ax, frame, 0, 33, 'blue', 40)
            self._draw_connections(ax, frame, self.hand_conns, 33, 21, 'red', 0.8)
            self._draw_points(ax, frame, 33, 21, 'red', 25)
            self._draw_connections(ax, frame, self.hand_conns, 54, 21, 'green', 0.8)
            self._draw_points(ax, frame, 54, 21, 'green', 25)
            ax.legend(['Поза', 'Левая рука', 'Правая рука'], loc='upper right')
            return []

        def on_key(event):
            nonlocal current_elev, current_azim, current_roll
            if event.key == ' ':
                if anim.running:
                    anim.pause()
                else:
                    anim.resume()
            elif event.key == 'r':
                current_elev = self.elev
                current_azim = self.azim
                current_roll = self.roll
                ax.view_init(elev=current_elev, azim=current_azim, roll=current_roll)

        print(f"Запуск анимации для: '{sentence.text}'")
        print(f"Всего кадров: {total_frames}, Знаков: {len(sentence.signs)}")
        anim = FuncAnimation(
            fig, update, frames=total_frames,
            init_func=init,
            interval=interval,
            blit=False,
            repeat=True,
            cache_frame_data=False
        )
        fig.canvas.mpl_connect('key_press_event', on_key)
        controls_text = """Управление:
Мышь: Вращение/Панорамирование/Масштаб | Пробел: Пауза/Воспроизведение | R: Сброс вида
1: Front | 2: Side | 3: Top | 4: Isometric"""
        fig.text(0.02, 0.02, controls_text, fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
        if save:
            print(f"Сохранение анимации в {save}...")
            writer_fps = min(sentence.fps, 15)
            anim.save(save, writer='pillow', fps=writer_fps, dpi=80, savefig_kwargs={'facecolor': 'white'})
            print(f"Анимация сохранена в: {save}")
        else:
            plt.show()
        return anim

    def animate_sign(self, sign: SignData, save: Optional[str] = None):
        """Анимация отдельного знака (для обратной совместимости)"""
        sentence = SentenceData(
            text=sign.word,
            signs=[sign],
            merged_keypoints=sign.keypoints,
            fps=sign.fps,
            word_boundaries=[(0, len(sign.keypoints) - 1)]
        )
        return self.animate_sentence(sentence, save)

    def plot_static_frame(self, kp: np.ndarray, idx: int = 0, title: str = "Кадр"):

        plt.style.use('default')
        fig = plt.figure(figsize=(12, 9))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111, projection='3d')
        self._setup_3d_axis(ax, f"{title} - Кадр {idx + 1}")
        frame = kp[idx]
        self._draw_connections(ax, frame, self.pose_conns, 0, 33, 'blue', 0.8)
        self._draw_points(ax, frame, 0, 33, 'blue', 40)
        self._draw_connections(ax, frame, self.hand_conns, 33, 21, 'red', 0.8)
        self._draw_points(ax, frame, 33, 21, 'red', 25)
        self._draw_connections(ax, frame, self.hand_conns, 54, 21, 'green', 0.8)
        self._draw_points(ax, frame, 54, 21, 'green', 25)
        ax.legend(['Поза', 'Левая рука', 'Правая рука'], loc='upper right')
        plt.tight_layout()
        plt.show()


# ==================== Основная функция ====================
def main():
    parser = argparse.ArgumentParser(description='Расширенный переводчик знакового языка с поддержкой предложений')
    parser.add_argument('cmd', choices=['add', 'translate', 'sentence', 'list', 'show', 'compare', 'export'],
                        # Добавляем 'export'
                        help='Команда для выполнения')

    parser.add_argument('--word', help='Слово для команд add/show')
    parser.add_argument('--video', help='Путь к видео для команды add')
    parser.add_argument('--synonyms', help='Синонимы (через запятую) для команды add')
    parser.add_argument('--text', help='Текст для перевода')
    parser.add_argument('--save-merged', action='store_true', help='Сохранить объединённое предложение')
    parser.add_argument('--pause-frames', type=int, default=10, help='Пауза между словами (в кадрах)')
    parser.add_argument('--transition-frames', type=int, default=5, help='Кадры для перехода между словами')
    parser.add_argument('--frame', type=int, default=0, help='Показать определённый кадр')
    parser.add_argument('--animate', action='store_true', help='Запустить анимацию')
    parser.add_argument('--save', help='Сохранить анимацию в файл')
    parser.add_argument('--elev', type=float, default=-76, help='Угол подъёма (-90 до 90)')
    parser.add_argument('--azim', type=float, default=-90, help='Азимут (0 до 360)')
    parser.add_argument('--export-format', choices=['avatar'], default='avatar',
                        help='Формат экспорта (avatar для 3D аватара)')
    parser.add_argument('--output', help='Путь для сохранения экспортированного файла')

    args = parser.parse_args()
    processor = Processor()
    visualizer = Visualizer(elev=args.elev, azim=args.azim)
    cmd = args.cmd
    if cmd == 'add':
        if not args.word or not args.video:
            parser.error("'add' требует --word и --video")
        synonyms = args.synonyms.split(',') if args.synonyms else []
        processor.add_sign(args.word, args.video, synonyms)
    elif cmd == 'list':
        signs = processor.db.list_all()
        print("Доступные знаки:")
        for w in signs:
            print(f" - {w}")
    elif cmd == 'translate':
        if not args.text:
            parser.error("'translate' требует --text")
        result = processor.translate(args.text)
        print("Переведённые знаки:")
        for sign in result:
            print(f"{sign.word} (база: {sign.base}, кадры: {len(sign.keypoints)})")

    elif cmd == 'export':
        if not args.text:
            parser.error("'export' требует --text")

        output_file = args.output
        if args.export_format == 'avatar':
            if ' ' in args.text:
                # Экспорт предложения
                try:
                    result = processor.export_sentence_animation(args.text, output_file)
                    print(f"Предложение успешно экспортировано в {result}")
                except Exception as e:
                    print(f"Ошибка при экспорте предложения: {e}")
            else:
                # Экспорт отдельного слова
                try:
                    result = processor.export_word_animation(args.text, output_file)
                    print(f"Слово успешно экспортировано в {result}")
                except Exception as e:
                    print(f"Ошибка при экспорте слова: {e}")

        # Добавляем экспорт в команду sentence

    elif cmd == 'sentence':

        if not args.text:
            parser.error("'sentence' требует --text")

        builder = processor.sentence_builder

        builder.pause_frames = args.pause_frames

        builder.transition_frames = args.transition_frames

        sentence = processor.translate_sentence(args.text, save_merged=args.save_merged)

        # Добавляем автоматический экспорт для 3D аватара

        if args.save:

            try:

                export_file = args.output or f"sentence_{args.text.replace(' ', '_')}_avatar.json"

                result = processor.export_sentence_animation(args.text, export_file)

                print(f"Анимация экспортирована для 3D аватара: {result}")

            except Exception as e:

                print(f"Ошибка при экспорте анимации: {e}")

        if args.animate or args.save:

            visualizer.animate_sentence(sentence, save=args.save)

        else:

            print(f"Объединённое предложение содержит {len(sentence.merged_keypoints)} кадров.")

            print(f"Границы знаков: {sentence.word_boundaries}")

        # Добавляем экспорт в команду show

    elif cmd == 'show':

        if not args.word:
            parser.error("'show' требует --word")

        sign = processor.db.get(args.word)

        if not sign:

            print(f"Знак для '{args.word}' не найден.")

        else:

            # Добавляем автоматический экспорт для 3D аватара

            if args.save:

                try:

                    export_file = args.output or f"{args.word}_avatar.json"

                    result = processor.export_word_animation(args.word, export_file)

                    print(f"Анимация экспортирована для 3D аватара: {result}")

                except Exception as e:

                    print(f"Ошибка при экспорте анимации: {e}")

            if args.animate or args.save:

                visualizer.animate_sign(sign, save=args.save)

            else:

                visualizer.plot_static_frame(sign.keypoints, idx=args.frame, title=sign.word)

    elif cmd == 'compare':
        tokens = re.findall(r"\w+", args.text.lower() if args.text else '')
        if len(tokens) < 2:
            parser.error("'compare' требует указать два слова в --text")
        s1 = processor.db.get(tokens[0])
        s2 = processor.db.get(tokens[1])
        if not s1 or not s2:
            print("Оба слова должны иметь соответствующие знаки в базе данных.")
        else:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121, projection='3d')
            visualizer._setup_3d_axis(ax1, f"{s1.word} - Кадр 1")
            visualizer._draw_connections(ax1, s1.keypoints[0], visualizer.pose_conns, 0, 33, 'blue')
            visualizer._draw_points(ax1, s1.keypoints[0], 0, 33, 'blue')
            ax2 = fig.add_subplot(122, projection='3d')
            visualizer._setup_3d_axis(ax2, f"{s2.word} - Кадр 1")
            visualizer._draw_connections(ax2, s2.keypoints[0], visualizer.pose_conns, 0, 33, 'blue')
            visualizer._draw_points(ax2, s2.keypoints[0], 0, 33, 'blue')
            plt.tight_layout()
            plt.show()
    else:
        parser.error(f"Неизвестная команда: {cmd}")


if __name__ == '__main__':
    main()
