import time
import numpy as np
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal


class ResponseSignals(QObject):
    callback_signal= pyqtSignal(str, dict)

class Run_Image(QRunnable):
    def __init__(self, all_img):
        super(QRunnable,self).__init__()
        self.all_img = all_img
        self.res = ResponseSignals()
        self.stop_flag=False

        
    def run(self):
        result = {'img': 0}
        img = np.zeros(4096)
        img = self.all_img
        result['img'] = img.reshape([64, 64])
        self.res.callback_signal.emit('doing', result)
   

class Run_Pixel(QRunnable):
    def __init__(self, all_img, all_t):
        super(QRunnable,self).__init__()
        self.all_img = all_img
        self.all_t = all_t
        self.res = ResponseSignals()

    def run(self):
        result = {'img': 0}
        img = np.zeros(4096)
        delay = 75
        for i in range(4096):
            if i > 170 :
                delay = 100
            time.sleep(self.all_t[i]*delay)
            img[i] = self.all_img[i]
            result['img'] = img.reshape([64, 64])
            result['i'] = i
            self.res.callback_signal.emit('doing', result)
      

class Run_Time(QRunnable):
    def __init__(self, t1,t2, all_t):
        super(QRunnable,self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.all_t = all_t
        self.res = ResponseSignals()
        
    def run(self):
        result = {}
        t1 = []
        t2 = []
        delay = 75
        for i in range(4096):
            if i > 170 :
                delay = 100
            time.sleep(self.all_t[i]*delay)
            t1.append(self.t1[i])
            t2.append(self.t2[i])
            result['time_cr'] = t1
            result['time_w'] = t2
            if i > 170:
                result['total_t1'] = np.sum(t1)
                result['total_t2'] = np.sum(t2)
            else :
                result['total_t1'] = 0
                result['total_t2'] = 0
            self.res.callback_signal.emit('doing', result)
       

class Run_MSE(QRunnable):
    def __init__(self, all_img, all_t):
        super(QRunnable,self).__init__()
        self.all_img = all_img
        self.all_t = all_t
        self.res = ResponseSignals()
        
    def run(self):
        result = {'mse': 0}
        mse = []
        delay = 75
        for i in range(4096):
            if i > 170 :
                delay = 100
            
            time.sleep(self.all_t[i] * delay)
            mse.append(self.all_img[i])
            result['mse'] = mse
            self.res.callback_signal.emit('doing', result)
      