import time


class CommandFilter:
    def __init__(self, cooldown=0.2, hold_time=0.15):
        self.last_command = "HOVER"
        self.last_change_time = time.time()
        self.cooldown = cooldown
        self.hold_time = hold_time

    def apply(self, new_command):
        now = time.time()

        # If same command ? allow
        if new_command == self.last_command:
            return self.last_command

        # If switching too fast ? ignore
        if now - self.last_change_time < self.cooldown:
            return self.last_command

        # Accept new command
        self.last_command = new_command
        self.last_change_time = now
        return new_command

    def fallback_hover(self, timeout, last_valid_time):
        if time.time() - last_valid_time > timeout:
            return "HOVER"
        return self.last_command
