import collections
import math

import mediapipe as mp


class GestureDetector:

    def __init__(self, history_size=9, min_votes=6):

        self.history = collections.deque(
            maxlen=history_size
        )

        self.min_votes = min_votes

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,

            # Keep Pi performance reasonable
            model_complexity=0,

            min_detection_confidence=0.60,
            min_tracking_confidence=0.65,
        )

    # -------------------------------------------------
    # GEOMETRY HELPERS
    # -------------------------------------------------

    @staticmethod
    def _distance(a, b):

        return math.sqrt(
            (a.x - b.x) ** 2
            + (a.y - b.y) ** 2
        )

    @staticmethod
    def _angle(a, b, c):
        """
        Angle ABC in degrees.
        """

        bax = a.x - b.x
        bay = a.y - b.y

        bcx = c.x - b.x
        bcy = c.y - b.y

        mag1 = math.sqrt(
            bax * bax + bay * bay
        )

        mag2 = math.sqrt(
            bcx * bcx + bcy * bcy
        )

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0

        dot = (
            bax * bcx
            + bay * bcy
        )

        cosine = dot / (mag1 * mag2)

        cosine = max(
            -1.0,
            min(1.0, cosine)
        )

        return math.degrees(
            math.acos(cosine)
        )

    # -------------------------------------------------
    # FINGER STATE
    # -------------------------------------------------

    def _finger_extended(
        self,
        lm,
        mcp,
        pip,
        dip,
        tip
    ):

        pip_angle = self._angle(
            lm[mcp],
            lm[pip],
            lm[dip],
        )

        dip_angle = self._angle(
            lm[pip],
            lm[dip],
            lm[tip],
        )

        wrist_to_tip = self._distance(
            lm[0],
            lm[tip],
        )

        wrist_to_pip = self._distance(
            lm[0],
            lm[pip],
        )

        straight_enough = (
            pip_angle > 150
            and dip_angle > 145
        )

        extended_enough = (
            wrist_to_tip
            >
            wrist_to_pip * 1.08
        )

        return (
            straight_enough
            and extended_enough
        )

    def _thumb_extended(self, lm):

        mcp_angle = self._angle(
            lm[1],
            lm[2],
            lm[3],
        )

        ip_angle = self._angle(
            lm[2],
            lm[3],
            lm[4],
        )

        wrist_to_tip = self._distance(
            lm[0],
            lm[4],
        )

        wrist_to_mcp = self._distance(
            lm[0],
            lm[2],
        )

        return (
            mcp_angle > 125
            and ip_angle > 140
            and wrist_to_tip
            > wrist_to_mcp * 1.10
        )

    def fingers_state(
        self,
        hand_landmarks,
        handedness_label=None,
    ):

        lm = hand_landmarks.landmark

        thumb = self._thumb_extended(lm)

        index = self._finger_extended(
            lm, 5, 6, 7, 8
        )

        middle = self._finger_extended(
            lm, 9, 10, 11, 12
        )

        ring = self._finger_extended(
            lm, 13, 14, 15, 16
        )

        pinky = self._finger_extended(
            lm, 17, 18, 19, 20
        )

        return [
            int(thumb),
            int(index),
            int(middle),
            int(ring),
            int(pinky),
        ]
        
    # -------------------------------------------------
    # GESTURE CLASSIFICATION
    # -------------------------------------------------

    def classify(self, fingers, lm):

        thumb, index, middle, ring, pinky = fingers

        palm_size = self._distance(
            lm[0],
            lm[9],
        )

        if palm_size < 1e-6:
            return "UNKNOWN"

        # ---------------------------------------------
        # Thumb direction
        # ---------------------------------------------

        thumb_dx = (
            lm[4].x - lm[2].x
        )

        thumb_dy = (
            lm[4].y - lm[2].y
        )

        thumb_vertical = (
            abs(thumb_dy)
            >
            abs(thumb_dx) * 0.9
        )

        strong_thumb_motion = (
            abs(thumb_dy)
            >
            palm_size * 0.45
        )

        thumb_up = (
            thumb == 1
            and thumb_vertical
            and strong_thumb_motion
            and thumb_dy < 0
        )

        thumb_down = (
            thumb == 1
            and thumb_vertical
            and strong_thumb_motion
            and thumb_dy > 0
        )

        long_fingers_closed = (
            index == 0
            and middle == 0
            and ring == 0
            and pinky == 0
        )

        # ---------------------------------------------
        # THUMB UP / DOWN
        # ---------------------------------------------

        if long_fingers_closed:

            if thumb_up:
                return "UP"

            if thumb_down:
                return "DOWN"

            # Proper fist:
            # all long fingers folded and thumb
            # not sticking strongly out
            if thumb == 0:
                return "BACKWARD"

            return "UNKNOWN"
            
        # ---------------------------------------------
        # OPEN PALM
        # Four long fingers clearly extended.
        # Thumb state intentionally not mandatory.
        # ---------------------------------------------

        if (
            index == 1
            and middle == 1
            and ring == 1
            and pinky == 1
        ):
            return "YAW_RIGHT"

        # ---------------------------------------------
        # L SHAPE
        # thumb + index extended
        # remaining fingers folded
        # ---------------------------------------------

        if (
            thumb == 1
            and index == 1
            and middle == 0
            and ring == 0
            and pinky == 0
        ):

            # Direction vectors
            tx = lm[4].x - lm[2].x
            ty = lm[4].y - lm[2].y

            ix = lm[8].x - lm[5].x
            iy = lm[8].y - lm[5].y

            tmag = math.sqrt(
                tx * tx + ty * ty
            )

            imag = math.sqrt(
                ix * ix + iy * iy
            )

            if tmag > 1e-6 and imag > 1e-6:

                dot = (
                    tx * ix
                    + ty * iy
                )

                cosine = dot / (
                    tmag * imag
                )

                cosine = max(
                    -1.0,
                    min(1.0, cosine)
                )

                separation = math.degrees(
                    math.acos(cosine)
                )

                if separation > 45:
                    return "YAW_LEFT"

            return "UNKNOWN"

        # ---------------------------------------------
        # VICTORY / TWO FINGERS
        # ---------------------------------------------

        if (
            index == 1
            and middle == 1
            and ring == 0
            and pinky == 0
        ):
            return "FORWARD"

        # ---------------------------------------------
        # INDEX ONLY
        # ---------------------------------------------

        if (
            index == 1
            and middle == 0
            and ring == 0
            and pinky == 0
            and thumb == 0
        ):
            return "RIGHT"

        # ---------------------------------------------
        # PINKY ONLY
        # ---------------------------------------------

        if (
            index == 0
            and middle == 0
            and ring == 0
            and pinky == 1
        ):
            return "LEFT"

        return "UNKNOWN"

    # -------------------------------------------------
    # TEMPORAL STABILITY FILTER
    # -------------------------------------------------

    def get_stable(self, gesture):

        self.history.append(gesture)

        if (
            len(self.history)
            <
            self.history.maxlen
        ):
            return "NONE"

        counts = collections.Counter(
            self.history
        )

        # UNKNOWN should never win a command
        counts.pop(
            "UNKNOWN",
            None
        )

        if not counts:
            return "NONE"

        gesture_name, votes = (
            counts.most_common(1)[0]
        )

        # Require enough votes
        if votes < self.min_votes:
            return "NONE"

        # Also require the recent frames to support it
        recent = list(
            self.history
        )[-3:]

        if recent.count(
            gesture_name
        ) < 2:
            return "NONE"

        return gesture_name

    # -------------------------------------------------
    # PROCESS FRAME
    # -------------------------------------------------

    def process(self, frame_rgb):

        results = self.hands.process(
            frame_rgb
        )

        if (
            not results.multi_hand_landmarks
        ):

            self.history.clear()

            return (
                "No Hand",
                "NONE",
                None,
                None,
            )

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )

        handedness = "Unknown"

        if results.multi_handedness:

            handedness = (
                results
                .multi_handedness[0]
                .classification[0]
                .label
            )

        fingers = self.fingers_state(
            hand_landmarks,
            handedness,
        )

        raw = self.classify(
            fingers,
            hand_landmarks.landmark,
        )

        stable = self.get_stable(
            raw
        )

        return (
            raw,
            stable,
            hand_landmarks,
            fingers,
        )
