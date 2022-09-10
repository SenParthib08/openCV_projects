import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_confidence,
                                         self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_res(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(imgRGB)

    def draw_hands(self, img, draw=True):
        res = self.find_res(img)
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_pos(self, img, hand_no=0, pos_no=8, draw=True):
        landmark_lst = list()
        height, width, channel = img.shape
        res = self.find_res(img)
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                c_x, c_y = int(lm.x * width), int(lm.y * height)
                landmark_lst.append([id, c_x, c_y])
                if draw and id == pos_no:
                    cv2.circle(img, (c_x, c_y), 15, (255, 0, 255), cv2.FILLED)
        return landmark_lst


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    tip_no = 8
    prev_time = 0
    curr_time = 0
    while True:
        success, img = cap.read()
        # img = cv2.flip(img, 1)
        img = detector.draw_hands(img)
        detector.find_pos(img, pos_no=tip_no)
        # landmark_lst = detector.find_pos(img, pos_no=tip_no)
        # if len(landmark_lst) != 0:
        #     print(landmark_lst[tip_no])
        #
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        #
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break


if __name__ == '__main__':
    main()
