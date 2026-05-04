VALID_COMMANDS = {
    "FORWARD",
    "BACKWARD",
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "YAW_LEFT",
    "YAW_RIGHT",
}


def gesture_to_command(stable_gesture):
    if stable_gesture in VALID_COMMANDS:
        return stable_gesture

    return None
