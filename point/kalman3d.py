import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from skimage.metrics import structural_similarity
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import math


# Initialization for system model.
# Matrix: A, H, Q, R, P_0
# Vector: x_0

H = np.array([[ 1,  0,  0,  0,  0,  0],
                           [ 0,  0,  1,  0,  0,  0],
                           [ 0,  0,  0,  0,  1,  0]])
Q = 1.0 * np.eye(6)
R = np.array([[50,  0, 0],
              [ 0, 50, 0],
              [ 0, 0, 50]])



class Kalman3D:
    def __init__(self, point, fps, start_frame, end_frame):
        self.point = point
        self.fps = fps
        self.dt = 1.0/fps
        self.T = 1
        self.esti = np.array([point.x[0], 0, point.y[0], 0, point.z[0], 0])
        self.P = 100.0*np.eye(6)
        self.A = np.array([[ 1, self.dt,  0,  0, 0, 0],
                            [ 0,  1,  0,  0, 0, 0],
                            [ 0,  0,  1, self.dt, 0, 0],
                            [ 0,  0,  0,  1, 0, 0],
                            [0, 0, 0, 0, 1, self.dt],
                            [0, 0, 0, 0, 0, 1]])
        self.x = []
        self.y = []
        self.z = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.v =[]
        self.frame = start_frame


    def prediction(self):

        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        self.esti = self.A @ self.esti
        self.P = self.A @ self.P @ self.A.T + Q

    def compute_gain(self):

        # Measurement Update (Correction)
        # ===============================
        # Compute the Kalman Gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ inv(S)

        return K

    def update(self, step, K):

        # Update the error covariance
        self.P = self.P - K @ H @ self.P

        # Update target coordinate
        self.x.append(float(self.esti[0]))
        self.y.append(float(self.esti[2]))
        self.z.append(float(self.esti[4]))
        self.vx.append(float(self.esti[1]))
        self.vy.append(float(self.esti[3]))
        self.vz.append(float(self.esti[5]))
        self.v.append(math.sqrt(float(self.esti[1]**2 + float(self.esti[3])**2 + float(self.esti[5])**2)))
        self.frame += 1


    def estimation(self, step, K):

        # Prediction the estimation
        if self.frame != self.point.frame_number[step]:
            idx = step
            
            while self.point.frame_number[step] != self.frame:
                
                # print('self.frame : {}, frame_number : {}'.format(self.frame, self.point.frame_number[step]))  
                
                Z = np.array([self.x[idx-1], self.y[idx-1], self.z[idx-1]])
                y = Z - H @ self.esti
                self.esti = self.esti + K @ y

                self.update(idx, K)

                idx += 1

                self.prediction()
                self.compute_gain()

        if self.point.x[step] == -1 and self.point.z[step] == -1:
            Z = np.array([self.x[step-1], self.y[step-1], self.z[step-1]])
        else :
            Z = np.array([self.point.x[step], self.point.y[step], self.point.z[step]])

    
        y = Z - H @ self.esti
        self.esti = self.esti + K @ y

        self.update(step, K)