import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import cvzone
import numpy as np

# Initialisation caméra et détecteur
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Paramètres globaux
keys = [["A", "Z", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["Q", "S", "D", "F", "G", "H", "J", "K", "L", "M"],
        ["W", "X", "C", "V", "B", "N", ",", ";", ":", "!"]]
message = ""
hoverTimeThreshold = 1  # Temps nécessaire pour activer une touche
alpha = 0.5  # Transparence

# Classe pour les boutons
class Button:
    def __init__(self, pos, text, size=[70, 70]):
        self.pos = pos
        self.size = size
        self.text = text

# Initialisation des boutons
buttonList = [Button([100 * j + 50, 100 * i + 50], key) for i, row in enumerate(keys) for j, key in enumerate(row)]
buttonList.extend([
    Button([1050, 50], "Del", size=[150, 70]),
    Button([1050, 150], "DelA", size=[150, 70]),
    Button([1050, 250], "spc", size=[150, 70]),
])

# Classe pour le rectangle de drag-and-drop
class DragRect:
    def __init__(self, posCenter, size=[1170, 210]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor[:2]

rect = DragRect([640, 450])  # Rectangle noir transparent

# Gestion du survol des boutons
hoverStartTime = {}

# Fonction pour dessiner les boutons
def drawAllButtons(img, buttons):
    overlay = img.copy()
    for button in buttons:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
        cvzone.cornerRect(overlay, (x, y, w, h), 20, rt=0)
        cv2.putText(overlay, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# Boucle principale
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)
    img = drawAllButtons(img, buttonList)

    currentTime = time.time()  # Temps actuel pour limiter les appels répétitifs

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        if lmList:
            cursor = lmList[8]  # Bout de l'index
            distance, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
            if distance < 50:
                rect.update(cursor)

            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                if x < cursor[0] < x + w and y < cursor[1] < y + h:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    if button.text not in hoverStartTime:
                        hoverStartTime[button.text] = currentTime
                    elif currentTime - hoverStartTime[button.text] > hoverTimeThreshold:
                        if button.text == "Del":
                            message = message[:-1]
                        elif button.text == "DelA":
                            message = ""
                        elif button.text == "spc":
                            message += " "
                        else:
                            message += button.text
                        hoverStartTime.clear()  # Réinitialiser après l'action
                else:
                    hoverStartTime.pop(button.text, None)

    # Dessiner le rectangle transparent
    cx, cy = rect.posCenter
    w, h = rect.size
    overlay = img.copy()
    cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 0, 0), cv2.FILLED)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.putText(img, message, (cx - w // 2 + 20, cy - h // 2 + 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
