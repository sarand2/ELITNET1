from time import localtime
import numpy as np
from numpy.linalg import inv
from sklearn import svm
import time


class HRPIAnalyzer:

    def __init__(self):
        try:
            self.one = 1
            self.N = 3                                 # fixed filter lag
            self.X = np.array([[0], [0], [0]])         # initial state vector
            self.Xhat = self.X                         # initial state estimate

            self.P = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])             # initial error covariance
            self.Ce = 1                                # measurement(observation) error(noise) covariance
            self.Cw = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]) * 0.0001   # process(state) error(noise) covariance

            self.PCol = np.zeros((3, 3, self.N + 1))
            self.PColOld = np.zeros((3, 3, self.N + 1))
            self.PSmooth = np.zeros((3, 3, self. N + 1))
            self.PSmoothOld = np.zeros((3, 3, self.N + 1))
            self.eye = np.eye(3, dtype=int)
            self.Support_vector_machine = svm.SVC(kernel='rbf')
            # svm.preloadTraining
        except Exception as e:
            self.logger.error('Failed to start HRPI analyzer ' + 'Exception message: ' + str(e))

    def KalmanFilterRun(self):
        # in normal run this is a while(true) kas 0.1 seconds
        start_time = time.time()
        while True:
            # H, Y = GetDataFromPcap() function in normal run (get 1 current and 3 previous timelapse values)
            # reikia gauti dabartini HRPI ir paskutinius 3 rodmenis
            H = np.matrix([Y[t-1], Y[t-2], Y[t-3]])
            # Form the Innovation vector.
            Inn = Y[t] - H.dot(self.Xhat)
            # Compute the covariance of the Innovation.
            S = H.dot(self.P).dot(H.T) + self.Ce
            # Form the Kalman Gain matrix.
            K = self.P.dot(H.T).dot(inv(S))
            # Update the state estimate.
            self.Xhat = self.Xhat + K.dot(Inn)
            XhatSmooth = self.Xhat
            # Compute the covariance of the estimation error.
            self.PColOld[:, :, 1] = self.P
            self.PSmoothOld[:, :, 1] = self.P
            self.P = (self.eye - K.dot(H)).dot(self.P) + self.Cw
            for i in range(1, self.N):
                KSmooth = self.PColOld[:, :, i].dot(H.T).dot(inv(S))
                self.PSmooth[:, :, i+1] = self.PSmoothOld[:, :, i] - self.PColOld[:, :, i].dot(H.T).dot(KSmooth.T)
                self.PCol[:, :, i+1] = self.ColOld[:, :, i].dot((self.eye - K.dot(H)).T)
                XhatSmooth = XhatSmooth + KSmooth.dot(Inn)
            self.PSmoothOld = self.PSmooth
            self.PColOld = self.PCol
            prediction = self.Support_vector_machine.predict(XhatSmooth)
            if prediction == 1:
                test = 1
                # ScriptCaller.Call()
            time.sleep(0.1 - ((time.time() - start_time) % 0.1))
