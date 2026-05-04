import time
import airsim


class AirSimAdapter:
    def __init__(self, ip, speed_cfg):
        self.client = airsim.MultirotorClient(ip=ip)
        self.speed = speed_cfg
        self.last_send_time = 0.0
        self.last_sent_display = "NONE"

    def connect_and_takeoff(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        print("Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        self.client.moveByVelocityBodyFrameAsync(0, 0, -0.5, 0.7).join()
        self.client.hoverAsync().join()

    def send_command(self, command):
        now = time.time()

        if now - self.last_send_time < self.speed["send_interval"]:
            return self.last_sent_display

        self.last_send_time = now

        duration = self.speed["command_duration"]

        if command == "HOVER":
            self.client.hoverAsync()
            self.last_sent_display = "HOVER"

        elif command == "FORWARD":
            self.client.moveByVelocityBodyFrameAsync(self.speed["forward"], 0, 0, duration)
            self.last_sent_display = "FORWARD"

        elif command == "BACKWARD":
            self.client.moveByVelocityBodyFrameAsync(self.speed["backward"], 0, 0, duration)
            self.last_sent_display = "BACKWARD"

        elif command == "RIGHT":
            self.client.moveByVelocityBodyFrameAsync(0, self.speed["right"], 0, duration)
            self.last_sent_display = "RIGHT"

        elif command == "LEFT":
            self.client.moveByVelocityBodyFrameAsync(0, self.speed["left"], 0, duration)
            self.last_sent_display = "LEFT"

        elif command == "UP":
            self.client.moveByVelocityBodyFrameAsync(0, 0, self.speed["up"], duration)
            self.last_sent_display = "UP"

        elif command == "DOWN":
            self.client.moveByVelocityBodyFrameAsync(0, 0, self.speed["down"], duration)
            self.last_sent_display = "DOWN"

        elif command == "YAW_LEFT":
            self.client.rotateByYawRateAsync(-self.speed["yaw_rate"], duration)
            self.last_sent_display = "YAW_LEFT"

        elif command == "YAW_RIGHT":
            self.client.rotateByYawRateAsync(self.speed["yaw_rate"], duration)
            self.last_sent_display = "YAW_RIGHT"

        return self.last_sent_display

    def land(self):
        self.client.landAsync().join()

    def cleanup(self):
        try:
            self.client.hoverAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass
