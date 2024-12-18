import sys
import os
import time
import cv2
import numpy as np
import mss
import datetime
import pyxhook
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QTextEdit, QWidget
from PyQt5.QtCore import Qt  # Импортируем Qt из PyQt5.QtCore
from PyQt5.QtCore import QTimer
from scipy.spatial.distance import pdist, squareform

from card_detector.card_detect import clear_and_extract_cards
from denomination_detector.denomination_detect import den_detect
from suit_detector.suit_detect import suit_detect
from probability_count.counter import print_equity_vs_all_types, get_all_equity

# Настройки
SCREENSHOT_FOLDER = "poker"
DETECTED_FOLDER = "detected_cards"
NUM_PLAYERS = 1

# Проверка и создание папок
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# Маппинг значений
SUIT_MAPPING = {"clubs": "c", "hearts": "h", "diamonds": "d", "spades": "s"}
DENOMINATION_MAPPING = {str(i): str(i) for i in range(2, 10)}
DENOMINATION_MAPPING.update({"1": "1", "10": "T", "J": "J", "Q": "Q", "K": "K", "A": "A"})


class PokerAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.hook_manager = None
        self.start_key_listener()
        self.num_players = NUM_PLAYERS

    def initUI(self):
        self.setWindowTitle("Анализатор Equity в покере")
        self.resize(600, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.output = QTextEdit(self)
        self.output.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.output)
        self.setLayout(layout)

    def take_screenshot(self):
        # Очистка папок перед новым скриншотом
        for folder in [SCREENSHOT_FOLDER, DETECTED_FOLDER]:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        screenshot_path = os.path.join(SCREENSHOT_FOLDER, f"screenshot_{timestamp}.png")

        with mss.mss() as sct:
            sct.shot(output=screenshot_path)
        self.process_image(screenshot_path)

    def process_image(self, image_path):
        self.output.append("\nЗапуск анализа карт...\n")

        coords = clear_and_extract_cards(input_folder=SCREENSHOT_FOLDER, output_folder=DETECTED_FOLDER)
        suits = suit_detect()
        denominations = den_detect()

        class Card:
            def __init__(self, coord, suit, denomination):
                self.suit = SUIT_MAPPING.get(suit, suit)
                self.denomination = DENOMINATION_MAPPING.get(denomination, denomination)
                self.center = ((coord[0] + coord[2]) / 2, (coord[1] + coord[3]) / 2)
                self.width = coord[2] - coord[0]

            def __repr__(self):
                return f"{self.denomination}{self.suit}"

        cards = [Card(coord, suit, denomination) for coord, suit, denomination in zip(coords, suits, denominations) if
                 suit != "not-card"]

        if not cards:
            self.output.append("Ошибка: карты не обнаружены. Проверьте скриншот.")
            return

        # Кластеризация
        centers = np.array([card.center for card in cards])
        distances = squareform(pdist(centers))
        card_width = np.mean([card.width for card in cards])
        threshold_distance = 1.5 * card_width

        clusters = []
        visited = set()

        def dfs(card_idx, cluster):
            if card_idx in visited:
                return
            visited.add(card_idx)
            cluster.append(cards[card_idx])
            for neighbor_idx, distance in enumerate(distances[card_idx]):
                if distance < threshold_distance:
                    dfs(neighbor_idx, cluster)

        for i in range(len(cards)):
            if i not in visited:
                cluster = []
                dfs(i, cluster)
                clusters.append(cluster)

        print(clusters)

        # Определение карт на столе и в руке
        hand_cards_clusters = [cluster for cluster in clusters if len(cluster) == 2]
        table_cards_clusters = [card for cluster in clusters if len(cluster) > 2 for card in cluster]

        if not hand_cards_clusters:
            self.output.append("Ошибка: карты в руке не обнаружены. Проверьте скриншот.")
            return

        if not table_cards_clusters:
            self.output.append("Предупреждение: карты на столе не обнаружены. Проверьте скриншот.")

        try:
            hand_cards = min(hand_cards_clusters, key=lambda x: x[0].center[1])
            table_cards = table_cards_clusters

            table = [f"{card.denomination}{card.suit}" for card in table_cards]
            hand = [f"{card.denomination}{card.suit}" for card in hand_cards]

            self.output.append("Карты на столе: " + ", ".join(table))
            self.output.append("Карты в руке: " + ", ".join(hand))
            self.output.append("Кол-во игроков: " + str(self.num_players))

            # Расчёт equity
            self.output.append("\nРассчет equity против всех типов оппонентов:\n")
            equities = get_all_equity(hand, table, self.num_players)

            for type, equity in equities.items():
                self.output.append(f"{type}: {equity:.2f}%")

        except ValueError as e:
            self.output.append(f"Ошибка: {str(e)}")

    def start_key_listener(self):
        def on_key_press(event):
            if event.Key == "space":
                QTimer.singleShot(0, self.output.clear)
                print("Пробел нажат! Делаю скриншот...")
                self.take_screenshot()
            elif event.Ascii in [ord(str(i)) for i in range(1, 10)]:
                self.num_players = int(chr(event.Ascii))
                self.output.append(f"Количество игроков изменено на {self.num_players}")

            elif event.Ascii == ord('0'):
                self.num_players = 10
                self.output.append("Количество игроков изменено на 10")

        self.hook_manager = pyxhook.HookManager()
        self.hook_manager.KeyDown = on_key_press
        self.hook_manager.HookKeyboard()
        print("Отслеживание нажатия пробела включено. Нажмите Ctrl+C для выхода.")
        self.hook_manager.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PokerAnalyzer()
    window.show()
    sys.exit(app.exec_())
