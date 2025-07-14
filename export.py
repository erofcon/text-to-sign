import json
import numpy as np

from sign_translator import SignData


def export_sentence_for_avatar(sentence_data: SignData, output_file="animation_data.json"):
    """
    Экспортировать данные предложения в формате JSON для анимации 3D аватара
    """
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


