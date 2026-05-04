import collections
import mediapipe as mp


class GestureDetector:
    def __init__(self, history_size=7, min_votes=5):
        self.history = collections.deque(maxlen=history_size)
        self.min_votes = min_votes

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

    def fingers_state(self, hand_landmarks, handedness_label):
        lm = hand_landmarks.landmark

        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y

        if handedness_label == "Right":
            thumb_open = lm[4].x < lm[3].x
        else:
            thumb_open = lm[4].x > lm[3].x

        return [
            int(thumb_open),
            int(index_up),
            int(middle_up),
            int(ring_up),
            int(pinky_up),
        ]

    def classify(self, fingers, lm):
        thumb, index, middle, ring, pinky = fingers

        others_closed = index == 0 and middle == 0 and ring == 0 and pinky == 0

        thumb_up = lm[4].y < lm[3].y
        thumb_down = lm[4].y > lm[3].y

        if others_closed and thumb_up:
            return "UP"

        if others_closed and thumb_down:
            return "DOWN"

        if fingers == [1, 1, 1, 1, 1]:
            return "YAW_RIGHT"

        if fingers == [0, 1, 1, 0, 0]:
            return "FORWARD"

        if fingers == [0, 0, 0, 0, 0]:
            return "BACKWARD"

        if fingers == [0, 1, 0, 0, 0]:
            return "RIGHT"

        if fingers == [0, 0, 0, 0, 1]:
            return "LEFT"

        if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
            return "YAW_LEFT"

        return "UNKNOWN"

    def get_stable(self, gesture):
        self.history.append(gesture)

        if len(self.history) < self.history.maxlen:
            return "NONE"

        most_common = collections.Counter(self.history).most_common(1)[0]

        if most_common[1] >= self.min_votes and most_common[0] != "UNKNOWN":
            return most_common[0]

        return "NONE"

    def process(self, frame_rgb):
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            self.history.clear()
            return "No Hand", "NONE", None, None

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label

        fingers = self.fingers_state(hand_landmarks, handedness)
        raw = self.classify(fingers, hand_landmarks.landmark)
        stable = self.get_stable(raw)

        return raw, stable, hand_landmarks, fingers
