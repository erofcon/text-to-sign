import os
import json
import argparse
import cv2
import numpy as np
from difflib import get_close_matches
from matplotlib import pyplot as plt

try:
    # Если хотите использовать эмбеддинги word2vec / FastText:
    from gensim.models import KeyedVectors
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

# ---------------------------------------------------
# 1. Собираем словарь: word -> путь к видео
# ---------------------------------------------------
VIDEO_DIR = "data/videos"   # структура: data/videos/слово1/word1.mp4
mapping = {}
for word in os.listdir(VIDEO_DIR):
    path = os.path.join(VIDEO_DIR, word)
    if not os.path.isdir(path):
        continue
    # пробуем найти видео в папке
    for fname in os.listdir(path):
        if fname.lower().endswith(('.mp4','.avi','.mov')):
            mapping[word.lower()] = os.path.join(path, fname)
            break

print(f"В базе: {len(mapping)} слов")

# ---------------------------------------------------
# 2. (Опционально) загружаем эмбеддинг
# ---------------------------------------------------
if EMBEDDING_AVAILABLE:
    print("Загружаем FastText-эмбеддинг для устойчивого поиска...")
    emb = KeyedVectors.load_word2vec_format("cc.ru.300.vec", binary=False)
else:
    emb = None

# ---------------------------------------------------
# 3. Нормализация ввода и поиск слова
# ---------------------------------------------------
def normalize(text: str) -> str:
    t = text.strip().lower()
    # тут можно добавить лемматизацию/стемминг
    return t

def find_best_match(query: str) -> str:
    q = normalize(query)
    if q in mapping:
        return q

    # 3.1. Попробуем близкий по строке
    close = get_close_matches(q, mapping.keys(), n=1, cutoff=0.7)
    if close:
        return close[0]

    # 3.2. Если есть эмбеддинг, ближайший по косинусу
    if emb and q in emb.key_to_index:
        sims = []
        for w in mapping.keys():
            if w in emb.key_to_index:
                sims.append((emb.similarity(q, w), w))
        best = max(sims, key=lambda x: x[0])
        return best[1]

    return None

# ---------------------------------------------------
# 4. Функция воспроизведения видео
# ---------------------------------------------------
def play_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Не удалось открыть видео:", path)
        return
    print("Нажмите Esc для выхода из окна")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Gesture", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Esc
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------
# 5. CLI
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", "-w", type=str, help="Слово для жеста")
    args = parser.parse_args()

    while True:
        if args.word:
            q = args.word
        else:
            q = input("Введите слово (или 'exit'): ")
            if q.lower() in ("exit","quit"):
                break

        key = find_best_match(q)
        if not key:
            print("Слово не найдено в базе.")
        else:
            print(f"Будет показан жест для слова: {key}")
            play_video(mapping[key])

        # если вызвано с --word, завершаем
        if args.word:
            break
