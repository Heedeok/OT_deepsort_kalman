from .point import Point

class Pointer:
    def __init__(self):
        self.points = []

    def new_point(self, idx):
        point = Point(idx)
        self.points.append(point)
