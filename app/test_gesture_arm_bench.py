import sys
import time
from pathlib import Path

import cv2
import yaml
import mediapipe as mp
from pymavlink import mavutil


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


from perception.camera import RaspberryPiCamera
from perception.gesture_detector import GestureDetector
from adapters.px4_offboard_adapter import PX4OffboardAdapter


with open(BASE_DIR / "config" / "real.yaml", "r") as f:
    cfg = yaml.safe_load(f)


ARM_HOLD_TIME = 1.5
AUTO_DISARM_TIME = 3.0


def send_arm_command(adapter, arm=True):

    value = 1.0 if arm else 0.0
    action = "ARM" if arm else "DISARM"

    print()
    print(f"Sending PX4 {action} request...")

    adapter.master.mav.command_long_send(
        adapter.master.target_system,
        adapter.master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        value,

        # IMPORTANT:
        # param2 = 0 means NORMAL arming.
        # No forced-arm bypass.
        0.0,

        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def wait_for_arm_state(adapter, target_armed, timeout=6):

    start = time.time()

    while time.time() - start < timeout:

        msg = adapter.master.recv_match(
            blocking=True,
            timeout=0.5,
        )

        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == "STATUSTEXT":
            try:
                print(f"PX4: {msg.text}")
            except Exception:
                pass

        elif msg_type == "COMMAND_ACK":

            try:
                if (
                    msg.command
                    ==
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
                ):
                    print(
                        f"ARM/DISARM ACK result: "
                        f"{msg.result}"
                    )
            except Exception:
                pass

        elif msg_type == "HEARTBEAT":

            armed = adapter.master.motors_armed()

            if bool(armed) == bool(target_armed):

                state = (
                    "ARMED"
                    if target_armed
                    else "DISARMED"
                )

                print(f"PX4 confirmed: {state}")

                return True

    return False

def main():

    print("==============================================")
    print(" DRONOSAUR GESTURE ARM BENCH TEST")
    print(" SHAKA HOLD -> ARM -> AUTO DISARM")
    print("==============================================")
    print()
    print("ABSOLUTE REQUIREMENTS:")
    print("  PROPS REMOVED")
    print("  AIRCRAFT SECURED")
    print("  NO OFFBOARD")
    print("  NO TAKEOFF")
    print("  NO THROTTLE COMMAND")
    print()

    # ------------------------------------------------
    # CONFIG SAFETY GATES
    # ------------------------------------------------

    if cfg["safety"].get(
        "allow_bench_arm_test",
        False
    ) is not True:

        raise RuntimeError(
            "SAFETY STOP: "
            "allow_bench_arm_test must be true."
        )

    if cfg["safety"].get(
        "allow_takeoff",
        False
    ) is not False:

        raise RuntimeError(
            "SAFETY STOP: allow_takeoff must be false."
        )

    # ------------------------------------------------
    # HUMAN CONFIRMATION
    # ------------------------------------------------

    confirmation = input(
        "Type PROPS_OFF to continue: "
    ).strip()

    if confirmation != "PROPS_OFF":

        print("Bench arm test cancelled.")
        return

    # ------------------------------------------------
    # CAMERA + GESTURE
    # ------------------------------------------------

    camera = RaspberryPiCamera(
        width=cfg["camera"]["width"],
        height=cfg["camera"]["height"],
        fps=cfg["camera"]["fps"],
    )

    detector = GestureDetector(
        history_size=cfg["gesture"]["history_size"],
        min_votes=cfg["gesture"]["min_votes"],
    )

    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # ------------------------------------------------
    # PIXHAWK
    # ------------------------------------------------

    adapter = PX4OffboardAdapter(
        device=cfg["connection"]["device"],
        baud=cfg["connection"]["baud"],
        send_rate_hz=10,
    )

    arm_hold_start = None
    arm_triggered = False

    try:

        camera.start()

        adapter.connect()

        print()
        print(f"Current mode : {adapter.get_mode()}")
        print(f"PX4 armed    : {adapter.is_armed()}")
        print()

        if adapter.is_armed():

            raise RuntimeError(
                "SAFETY STOP: vehicle already armed."
            )

        print("ARM gesture = SHAKA")
        print("Thumb + pinky extended")
        print("Index + middle + ring folded")
        print()
        print(
            f"Hold continuously for "
            f"{ARM_HOLD_TIME:.1f} seconds."
        )
        print()
        print("Press q to quit.")
        print()

        while True:

            ok, frame = camera.read()

            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            (
                raw,
                stable,
                landmarks,
                fingers,
            ) = detector.process(rgb)

            if landmarks is not None:

                mp_draw.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

            # ----------------------------------------
            # DEDICATED ARM GESTURE
            #
            # fingers =
            # [thumb,index,middle,ring,pinky]
            # ----------------------------------------

            shaka = (
                fingers
                ==
                [1, 0, 0, 0, 1]
            )

            progress = 0.0

            if (
                shaka
                and not arm_triggered
            ):

                if arm_hold_start is None:
                    arm_hold_start = time.time()

                elapsed = (
                    time.time()
                    - arm_hold_start
                )

                progress = min(
                    elapsed / ARM_HOLD_TIME,
                    1.0
                )

                if elapsed >= ARM_HOLD_TIME:

                    arm_triggered = True

                    print()
                    print(
                        "SHAKA HOLD CONFIRMED."
                    )

                    send_arm_command(
                        adapter,
                        arm=True,
                    )

                    armed_ok = wait_for_arm_state(
                        adapter,
                        target_armed=True,
                        timeout=6,
                    )

                    if not armed_ok:

                        print()
                        print(
                            "PX4 DID NOT ARM."
                        )
                        print(
                            "No force-arm will be attempted."
                        )

                        break

                    print()
                    print(
                        "================================"
                    )
                    print(
                        " GESTURE ARM TEST PASSED"
                    )
                    print(
                        "================================"
                    )
                    print()
                    print(
                        f"Auto-disarming in "
                        f"{AUTO_DISARM_TIME:.0f} seconds..."
                    )

                    time.sleep(
                        AUTO_DISARM_TIME
                    )

                    send_arm_command(
                        adapter,
                        arm=False,
                    )

                    disarmed_ok = wait_for_arm_state(
                        adapter,
                        target_armed=False,
                        timeout=6,
                    )

                    if disarmed_ok:

                        print()
                        print(
                            "AUTO DISARM CONFIRMED."
                        )

                    else:

                        print()
                        print(
                            "WARNING: DISARM NOT "
                            "CONFIRMED."
                        )

                    break

            else:

                # Any interruption resets arm hold
                if not shaka:
                    arm_hold_start = None

            # ----------------------------------------
            # DISPLAY
            # ----------------------------------------

            gesture_text = (
                "SHAKA - HOLD TO ARM"
                if shaka
                else "NO ARM GESTURE"
            )

            cv2.putText(
                frame,
                gesture_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Fingers: {fingers}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"PX4 Armed: {adapter.is_armed()}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"ARM hold: "
                    f"{progress * 100:.0f}%"
                ),
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 180, 255),
                2,
            )

            cv2.putText(
                frame,
                "PROPS OFF | BENCH ARM TEST",
                (
                    20,
                    frame.shape[0] - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
            )

            cv2.imshow(
                "Dronosaur Gesture ARM Bench",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        print()
        print("Cleaning up ARM bench test...")

        # Defensive disarm request on exit.
        try:

            if (
                adapter.master is not None
                and adapter.is_armed()
            ):

                print(
                    "Vehicle still reports ARMED."
                )

                send_arm_command(
                    adapter,
                    arm=False,
                )

                wait_for_arm_state(
                    adapter,
                    False,
                    timeout=5,
                )

        except Exception as e:

            print(
                f"Cleanup disarm warning: {e}"
            )

        try:
            camera.stop()
        except Exception:
            pass

        try:
            detector.hands.close()
        except Exception:
            pass

        try:
            adapter.cleanup()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print("Finished.")


if __name__ == "__main__":
    main()
