import math

import numpy as np


class Camera:
    def __init__(self, position=(8.0, 6.0, 9.0), yaw=-math.pi / 4, pitch=-0.7):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = 0.0
        self.speed = 7.0
        self.sensitivity = 0.0025

    def forward(self):
        return np.array(
            [
                math.cos(self.pitch) * math.sin(self.yaw),
                math.sin(self.pitch),
                -math.cos(self.pitch) * math.cos(self.yaw),
            ],
            dtype=np.float32,
        )

    def right(self):
        return np.array([math.cos(self.yaw), 0.0, math.sin(self.yaw)], dtype=np.float32)

    def up(self):
        return np.cross(self.right(), self.forward())

    def view_matrix(self):
        target = self.position + self.forward()
        forward = self.forward()
        right = self.right()
        up = np.cross(right, forward)
        if self.roll:
            up = right * math.sin(self.roll) + up * math.cos(self.roll)
        return self.look_at(self.position, target, up)

    @staticmethod
    def look_at(eye, center, up):
        eye = np.asarray(eye, dtype=np.float32)
        center = np.asarray(center, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)

        zaxis = eye - center
        zaxis /= np.linalg.norm(zaxis) + 1e-8
        xaxis = np.cross(up, zaxis)
        xaxis /= np.linalg.norm(xaxis) + 1e-8
        yaxis = np.cross(zaxis, xaxis)

        return np.array(
            [
                [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
                [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
                [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def projection_matrix(self, width, height, fov=60.0, near=0.1, far=200.0):
        aspect = max(width / max(height, 1), 0.01)
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        return np.array(
            [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
