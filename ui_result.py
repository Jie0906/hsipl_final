import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy,QMessageBox
from PyQt5.QtCore import QCoreApplication,Qt
from result import Ui_MainWindow
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.Qt import QThreadPool
from method import Run_Pixel,Run_Image,Run_MSE ,Run_Time

filename =  r"res.npz"
res = np.load(filename, allow_pickle=True)
R_RXD = res['r_rxd']    
CR_RXD = res['cr_rxd']
RT_CR_RXD = res['rt_cr_rxd']
T_CR =res['t_cr']
t_rt_cr =res['t_rt_cr']
k_rxd = res['k_rxd']    
ck_rxd = res['ck_rxd']
rt_ck_rxd = res['rt_ck_rxd']
t_ck =res['t_ck']
t_rt_ck =res['t_rt_ck']
mse_r = res['mse_r']
mse_k = res['mse_k']

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)    
        self.fig_rxd = FullFigure(width=5, height=4, dpi=100)
        self.fig_rxd.axes.axis('off')
        self.ui.verticalLayout.addWidget(self.fig_rxd)
        
        self.fig_c_rxd = FullFigure(width=5, height=4, dpi=100)
        self.fig_c_rxd.axes.axis('off')
        self.ui.verticalLayout_2.addWidget(self.fig_c_rxd)

        
        self.fig_woodbury = FullFigure(width=5, height=4, dpi=100)
        self.fig_woodbury.axes.axis('off')
        self.ui.verticalLayout_3.addWidget(self.fig_woodbury)

        
        self.ui.pushButton_2.clicked.connect(self.pushButton_2_Click)
        self.ui.pushButton_3.clicked.connect(self.pushButton_3_Click)
        
        self.fig_time = InwardFigure(width=5, height=4, dpi=100)
        self.fig_time.axes.set_xlim(0, 4096)
        self.fig_time.axes.set_title(' Woodbury & Cost_time Casual')
        self.fig_time.axes.set_ylabel('Time(sec)')
        self.ui.verticalLayout_4.addWidget(self.fig_time)
        
        self.fig_mse = InwardFigure(width=5, height=4, dpi=100)
        self.fig_mse.axes.set_xlim(0, 4096)
        self.fig_mse.axes.set_title(' Woodbury & MSE Casual')
        self.fig_mse.axes.set_ylabel('Time(sec)')
        self.ui.verticalLayout_5.addWidget(self.fig_mse)
        
        self.show()
   
    def pushButton_2_Click(self):
        max_time = np.maximum(T_CR,t_rt_cr)
        
        do_r = Run_Image(R_RXD)
        do_r.res.callback_signal.connect(self.RShow)
        QThreadPool.globalInstance().start(do_r) 

        do_cr = Run_Pixel(CR_RXD,T_CR)
        do_cr.res.callback_signal.connect(self.CRShow)
        QThreadPool.globalInstance().start(do_cr) 
        
        do_w = Run_Pixel(RT_CR_RXD,t_rt_cr)
        do_w.res.callback_signal.connect(self.WShow)
        QThreadPool.globalInstance().start(do_w) 
      
        do_time = Run_Time(T_CR,t_rt_cr,max_time)
        do_time.res.callback_signal.connect(self.timeShow)
        QThreadPool.globalInstance().start(do_time) 
      
        do_mse = Run_MSE(mse_r,max_time)
        do_mse.res.callback_signal.connect(self.mseShow)
        QThreadPool.globalInstance().start(do_mse) 
        

    def pushButton_3_Click(self):
        max_time = np.maximum(t_ck,t_rt_ck)
        
        do_r = Run_Image(k_rxd)
        do_r.res.callback_signal.connect(self.RShow)
        QThreadPool.globalInstance().start(do_r) 

        do_cr = Run_Pixel(ck_rxd,t_ck)
        do_cr.res.callback_signal.connect(self.CRShow)
        QThreadPool.globalInstance().start(do_cr) 
        
        do_w = Run_Pixel(rt_ck_rxd,t_rt_ck)
        do_w.res.callback_signal.connect(self.WShow)
        QThreadPool.globalInstance().start(do_w) 
      
        do_time = Run_Time(t_ck,t_rt_ck,max_time)
        do_time.res.callback_signal.connect(self.timeShow)
        QThreadPool.globalInstance().start(do_time) 
      
        do_mse = Run_MSE(mse_k,max_time)
        do_mse.res.callback_signal.connect(self.mseShow)
        QThreadPool.globalInstance().start(do_mse)

    def RShow(self, msg, result):
        img = result['img']

        self.fig_rxd.axes.clear()
        self.fig_rxd.axes.axis('off')
        self.fig_rxd.axes.imshow(img, cmap='gray')   

        self.fig_rxd.draw()

        self.ui.verticalLayout.addWidget(self.fig_rxd)

    def CRShow(self, msg, result):
        img = result['img']
        i = result['i']
        percent = np.around((i-175)/(4095-175)*100,2)
        if i<=170:
            percent = 0
        self.fig_c_rxd.axes.clear()
        self.fig_c_rxd.axes.axis('off')
        self.fig_c_rxd.axes.imshow(img, cmap='gray')    
        self.fig_c_rxd.draw()
        self.ui.label_5.setText("Casual : " + str(percent) + "%")
        self.ui.verticalLayout_3.addWidget(self.fig_c_rxd) 

    def WShow(self, msg, result):
        img = result['img']
        i = result['i']
        percent = np.around((i-175)/(4095-175)*100,2)
        if i<=175:
            percent = 0            
        self.fig_woodbury.axes.clear()
        self.fig_woodbury.axes.axis('off')
        self.fig_woodbury.axes.imshow(img, cmap='gray')  
        self.ui.label_4.setText("Woodbury : " + str(percent) + "%")
        self.fig_woodbury.draw()
        self.ui.verticalLayout_2.addWidget(self.fig_woodbury) 
    
    def timeShow(self,msg,result):
        time_cr=result['time_cr']
        time_w=result['time_w']
        total_time_cr = result['total_t1']
        total_time_w = result['total_t2']
        total_time_cr = np.around(total_time_cr, 5)
        total_time_w = np.around(total_time_w, 5)
        self.fig_time.axes.clear()
        self.fig_time.axes.set_title('Woodbury & Cost time Casual')
        self.fig_time.axes.set_ylabel('Time(sec)')
        self.fig_time.axes.plot(time_cr, 'y', label='Casual : '+str(total_time_cr))
        self.fig_time.axes.plot(time_w, 'b', label='Woodbury : '+str(total_time_w))
        self.fig_time.axes.legend()
        self.fig_time.draw()
        self.ui.verticalLayout_4.addWidget(self.fig_time) 
            
    def mseShow(self,msg,result):
        mse = result['mse']

        self.fig_mse.axes.clear()
        self.fig_mse.axes.set_title('Woodbury & MSE Casual')
        self.fig_mse.axes.set_ylabel('Time(sec)')
        self.fig_mse.axes.plot(mse, 'g')
        self.fig_mse.draw()
        self.ui.verticalLayout_5.addWidget(self.fig_mse)

class FullFigure(FigureCanvas):
    def __init__(self,parent=None,width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        self.axes = self.fig.add_subplot(111)
        self.axes.margins(0, 0)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
class InwardFigure(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        self.axes.margins(0, 0)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        
finalResult = QCoreApplication.instance()
if finalResult is None:
    finalResult = QApplication(sys.argv)
demo= AppWindow()
demo.show()
sys.exit(finalResult.exec_())
