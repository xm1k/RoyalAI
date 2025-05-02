import sys
import os
import datetime
import time

import cv2
import numpy as np
import mss
import pyxhook
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QTextEdit, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QTextCursor, QFont, QColor, QTextCharFormat
from scipy.spatial.distance import pdist, squareform
from card_detector.card_detect import clear_and_extract_cards
from denomination_detector.denomination_detect import den_detect
from suit_detector.suit_detect import suit_detect
from probability_count.counter import get_all_equity
from PyQt5.QtGui import QFontDatabase

SCREENSHOT_FOLDER = "poker"
DETECTED_FOLDER = "detected_cards"
NUM_PLAYERS = 1

os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

SUIT_MAPPING = {"clubs": "c", "hearts": "h", "diamonds": "d", "spades": "s"}
DENOMINATION_MAPPING = {str(i): str(i) for i in range(2, 10)}
DENOMINATION_MAPPING.update({"1": "1", "10": "T", "J": "J", "Q": "Q", "K": "K", "A": "A"})
SUIT_COLORS = {"c": "#32cd32", "h": "#ff0000", "d": "#87cefa", "s": "#808080"}

class PokerAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.hook_manager = None
        self.start_key_listener()
        self.num_players = NUM_PLAYERS
        self.hand_cards = []  # Список для хранения карт игрока
        self.table_cards = []

    def initUI(self):
        self.setWindowTitle("Анализатор Equity в покере")
        self.resize(400, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        self.setStyleSheet("background-color: #000000;")

        # Добавление собственного шрифта
        font_id = QFontDatabase.addApplicationFont("nothing.ttf")
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        font = QFont(font_family, 24, QFont.Bold)

        # Добавляем надпись "RoyalAI" сверху в центре окна
        self.title = QLabel("RoyalAI", self)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(font)
        self.title.setStyleSheet(
            "color: rgba(255, 0, 0, 0.7); padding: 5px; border: 2px dashed rgba(255, 0, 0, 0.5); background-color: #000000;")

        self.output = QTextEdit(self)
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 12))
        border_style = "border: 2px dashed rgba(255, 0, 0, 0.5);"

        self.setStyleSheet(f"background-color: #000000;")

        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 12))
        self.output.setStyleSheet(
            f"background-color: #000000; color: #dcdcdc; padding: 10px; border: none; {border_style}"
        )
        self.output.setFrameShape(QTextEdit.NoFrame)  # Убираем рамку вокруг QTextEdit

        self.setStyleSheet(f"background-color: #000000; {border_style}")

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.output)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        self.move_to_bottom_right()

    def move_to_bottom_right(self):
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        x = screen_geometry.right() - self.width()
        y = screen_geometry.bottom() - self.height()
        self.move(x, y)

    def append_colored_text(self, text, color):
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)

        format = QTextCharFormat()
        format.setForeground(QColor(color))
        format.setFontPointSize(12)
        format.setFont(QFont("Consolas", 12))

        block_format = cursor.blockFormat()
        block_format.setAlignment(Qt.AlignCenter)
        cursor.setBlockFormat(block_format)

        cursor.insertText(text + "\n", format)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def append_card_text(self, cards):
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)

        for card in cards:
            color = SUIT_COLORS.get(card[-1], "#dcdcdc")
            format = QTextCharFormat()
            format.setForeground(QColor(color))
            format.setFontPointSize(12)
            format.setFont(QFont("Consolas", 12))

            cursor.insertText(card + " ", format)

        cursor.insertText("\n")
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def take_screenshot(self):
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

        cards = [Card(coord, suit, denomination) for coord, suit, denomination in zip(coords, suits, denominations) if suit != "not-card"]

        if not cards:
            self.append_colored_text("Ошибка: карты не обнаружены. Проверьте скриншот.", "#ff0000")
            return

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

        hand_cards_clusters = [cluster for cluster in clusters if len(cluster) == 2]
        table_cards_clusters = [card for cluster in clusters if len(cluster) > 2 for card in cluster]

        if not hand_cards_clusters:
            self.append_colored_text("Ошибка: карты в руке не обнаружены. Проверьте скриншот.", "#ff0000")
            return

        if not table_cards_clusters:
            self.append_colored_text("Предупреждение: карты на столе не обнаружены.", "#ffa500")

        try:
            hand_cards = min(hand_cards_clusters, key=lambda x: x[0].center[1])
            table_cards = table_cards_clusters

            self.hand_cards = hand_cards  # Сохраняем карты игрока
            self.table_cards = table_cards  # Сохраняем карты на столе

            table = [f"{card.denomination}{card.suit}" for card in table_cards]
            hand = [f"{card.denomination}{card.suit}" for card in hand_cards]

            time.sleep(0.2)
            self.append_colored_text("", "#ffffff")
            self.append_card_text(table)
            self.append_colored_text("", "#ffffff")
            self.append_card_text(hand)

            self.append_colored_text("", "#ffffff")
            self.append_colored_text("Соперников: " + str(self.num_players), "#ffffff")

            self.append_colored_text("\nEquity:\n", "#ffffff")
            equities = get_all_equity(hand, table, self.num_players)

            for type, equity in equities.items():
                self.append_colored_text(f"{type}: {equity:.2f}%", "#1e90ff")


        except ValueError as e:
            self.append_colored_text(f"Ошибка: {str(e)}", "#ff0000")

    def start_key_listener(self):
        def on_key_press(event):
            if event.Key == "space":
                QTimer.singleShot(0, self.output.clear)
                self.take_screenshot()
            elif event.Ascii in [ord(str(i)) for i in range(1, 10)]:
                self.num_players = int(chr(event.Ascii))
                self.append_colored_text("", "#ffffff")
                self.append_colored_text(f"Соперников: {self.num_players}", "#1e90ff")
                if self.hand_cards and self.table_cards:
                    self.recalculate_equity()

            elif event.Ascii == ord('0'):
                self.num_players = 10
                self.append_colored_text("","#ffffff")
                self.append_colored_text("Соперников: 10", "#ffffff")
                if self.hand_cards and self.table_cards:
                    self.recalculate_equity()

        self.hook_manager = pyxhook.HookManager()
        self.hook_manager.KeyDown = on_key_press
        self.hook_manager.HookKeyboard()
        self.hook_manager.start()

    def recalculate_equity(self):
        hand = [f"{card.denomination}{card.suit}" for card in self.hand_cards]
        table = [f"{card.denomination}{card.suit}" for card in self.table_cards]

        try:
            QTimer.singleShot(0, self.output.clear)
            hand_cards = self.hand_cards
            table_cards = self.table_cards

            self.hand_cards = hand_cards  # Сохраняем карты игрока
            self.table_cards = table_cards  # Сохраняем карты на столе


            table = [f"{card.denomination}{card.suit}" for card in table_cards]
            hand = [f"{card.denomination}{card.suit}" for card in hand_cards]

            time.sleep(0.5)
            self.append_colored_text("", "#ffffff")
            self.append_card_text(table)
            self.append_colored_text("", "#ffffff")
            self.append_card_text(hand)

            self.append_colored_text("", "#ffffff")
            self.append_colored_text("Соперников: " + str(self.num_players), "#ffffff")

            self.append_colored_text("\nEquity:\n", "#ffffff")
            equities = get_all_equity(hand, table, self.num_players)

            for type, equity in equities.items():
                self.append_colored_text(f"{type}: {equity:.2f}%", "#1e90ff")
        except:
            pass
        # self.append_colored_text("\nEquity пересчитывается...\n", "#ffffff")
        # equities = get_all_equity(hand, table, self.num_players)
        #
        # for type, equity in equities.items():
        #     self.append_colored_text(f"{type}: {equity:.2f}%", "#1e90ff")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PokerAnalyzer()
    window.show()
    sys.exit(app.exec_())
