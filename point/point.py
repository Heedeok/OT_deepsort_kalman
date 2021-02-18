from .kalman3d import Kalman3D
import math

class Point:
    def __init__(self, idx):
        self.idx = idx
        self.frame_number = []
        self.x =[]
        self.y =[]
        self.z =[]
        self.kf = None
        self.v = []

    def initiate_kalman(self, fps=60):

        self.kf = Kalman3D(self, fps, start_frame=self.frame_number[0], end_frame=self.frame_number[-1])

        for i in range(len(self.frame_number)):
            self.kf.prediction()
            K = self.kf.compute_gain()
            self.kf.estimation(i,K)
    
    def calculate_velocity(self, fps):

        frame = self.frame_number[0]

        for i in range(len(self.frame_number)):
            if i == 0:
                v = 0
            else:
                while frame != self.frame_number[i]:
                    self.v.append(self.v[i-1])
                    frame += 1
                
                dist = math.sqrt((self.x[i]-self.x[i-1])**2 + (self.y[i]-self.y[i-1])**2 + (self.z[i]-self.z[i-1])**2)

                v = dist/(1./fps)
            
            self.v.append(v)

            frame += 1