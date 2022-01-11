import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def woodBury (temp,X,Y):
    XT = np.transpose(X)
    YT = np.transpose(Y) 

    return temp - [(temp@XT) @ (YT @ temp)/(1 + YT @ temp @ XT)]

def R (him):
    
    N = him.shape[0]
    ri = np.reshape(him,(N,him.shape[1]))
    rit = np.transpose(ri)
    R = np.dot(rit,(ri))/N
    
    return R


def K (him):

    N = him.shape[0]
    ri = np.reshape(him,(N,him.shape[1]))
    rit = np.transpose(ri)
    u = (np.mean(rit, 1))
    K = np.dot(np.transpose(ri-u),(ri-u))/N

    return u,K




def R_RXD(him):

    N = him.shape[0] #64*64 = 4096
    r = R(him)
    invr = np.linalg.inv(r)
    ri = np.reshape(him,(N,him.shape[1]))

    R_RXD = np.zeros(N)
    for i in range(N):
        R_RXD[i] = ri[i] @ invr @ np.reshape(ri,(him.shape[0],1))


    return invr, R_RXD   


def K_RXD (him):

    N = him.shape[0] #64*64 = 4096
    u,k = K(him)
    invk = np.linalg.inv(k)
    ri = np.reshape(him,(N,him.shape[1]))

    K_RXD = np.zeros(N)
    for i in range(N):
        ru = ri[i,:]-u
        K_RXD[i] = ru @ invk @ np.reshape(ru,(him.shape[0],1))

    return invk, u, K_RXD




def CR_RXD(ri,himImg):

    bands = ri.shape[0]
    r = R(himImg)
    invr = np.linalg.inv(r)
    CR_RXD = ri @ invr @ np.reshape(ri,(1,bands))

    return CR_RXD
    
    


def CK_RXD(ri,himImg):

    bands = ri.shape[0]
    u,k = K(himImg)
    invk = np.linalg.inv(k)
    ru = ri - u
    CK_RXD = ru @ invk @ np.reshape(ru,(1,bands))

    return u, CK_RXD


def RT_CR_RXD(ri,n,pre_R):
    bands = ri.shape[1] 
    rt = np.reshape(ri,(bands,1))
    a = np.linalg.inv(((n-1)/n)*pre_R)
    b = c = 1/np.sqrt(n)*ri
    invR = woodBury(a, b, c)
    img = ri @ invR @ rt
    return invR, img


def RT_CK_RXD(ri,n,pre_K,pre_u):
    bands = ri.shape[1]
    ri = np.reshape(ri,(1,ri.shape[1]))
    u = (1-1/n) * pre_u + (1/n)*ri
    a = np.linalg.inv((1-1/n)*pre_K)
    b = c = (np.sqrt((n-1))/n)*(pre_u-ri) 
    invK = woodBury(a, b, c)
    ru = ri-u
    pixel = ru @ invK @ np.reshape(ru,(bands,1))

    return invK, u, pixel

   
def R_RXD_PLOT(him):
    
    x,img = R_RXD(him)
    tempImg = np.zeros((64, 64))
    tempImg = np.reshape(img,(64,64))
    plt.figure()
    plt.title("R-RXD")
    plt.axis("off")
    plt.imshow(tempImg,cmap='gray')
    return img

def K_RXD_PLOT(him):

    x,y,img = K_RXD(him)
    tempImg = np.zeros((64, 64))
    tempImg = np.reshape(img,(64,64))
    
    plt.figure()
    plt.title("K-RXD")
    plt.axis("off")
    plt.imshow(tempImg,cmap='gray')
    return img


def FUN_RXD_PLOT(him,status,fun):
    str = ''
    N = him.shape[0] #4096
    ln = int(np.sqrt(N))
    temparr = []
    Time = []
    tempImg = np.zeros((ln,ln))
    rtImg = np.zeros(N)
    
    if status == 1:
        if fun == R_RXD:
            str = 'R_RXD'
            plt.figure(str)
            plt.title(str)
            plt.axis("off")
            xrow = plt.subplot2grid((1, 1), (0, 0))
            xrow.set_title(str)
            xrow.axis("off")
            img = xrow.imshow(np.zeros((64,64)))
        if fun == K_RXD:
            str = 'K_RXD'
            plt.figure(str)
            plt.title(str)
            plt.axis("off")
            xrow = plt.subplot2grid((1, 1), (0, 0))
            xrow.axis("off")
            xrow.set_title(str)
            img = xrow.imshow(np.zeros((64,64)))
        
   
    for i in range (N):      
        if i <= startPixel :
            timeA = time.perf_counter()
            timeB = time.perf_counter() - timeA
            rtImg[0:i+1] = fun(him[0:i+1])
            temparr.append(rtImg)
        if i > startPixel :
            timeA = time.perf_counter()
            timeB = time.perf_counter() - timeA
            rtImg[i] = fun(him[i],him[0:i+1])
            temparr.append(rtImg)
        if status == 1:
            tempImg = np.reshape(rtImg,(ln,ln))
            img.set_data(tempImg)
            vmin=np.min(tempImg)
            vmax=np.max(tempImg)
            img.set_clim(vmin, vmax)
            plt.pause(1e-60)
            
        Time.append(timeB)
        
    return Time,rtImg


def RT_FUN_RXD_PLOT(him, status,fun,fun2):
    str = ''
    N = him.shape[0]
    ln = np.sqrt(N)
    ln = int(ln)
    temparr=[]
    Time = []
    tempImg = np.zeros((ln,ln))
    rtImg = np.zeros(N)
    

    if status == 1:
        if fun2 == RT_CR_RXD:
            str = "RT_CR_RXD"
            plt.figure(str)
            xrow = plt.subplot2grid((1, 1), (0, 0))
            xrow.axis("off")
            xrow.set_title(str)
            img = xrow.imshow(np.zeros((64,64)))
        if fun2 == RT_CK_RXD:
            str = "RT_CK_RXD"
            plt.figure(str)
            xrow = plt.subplot2grid((1, 1), (0, 0))
            xrow.axis("off")
            xrow.set_title(str)
            img = xrow.imshow(np.zeros((64,64)))
            
 
    for i in range (N):
        
        if i <= startPixel :
            timeA = time.perf_counter()
            timeB = time.perf_counter() - timeA
            pre_R,rtImg[:i+1]=fun(him[:i+1])
            temparr.append(pre_R,rtImg)
           
        if i > startPixel :
            pre_R = np.linalg.inv(pre_R)
            timeA = time.perf_counter()
            timeB = time.perf_counter() - timeA
            pre_R,rtImg[i:i+1] = fun2(him[i:i+1],i+1 ,pre_R)        
            temparr.append(pre_R,rtImg)

        if status == 1:
            tempImg = np.reshape(rtImg,(ln,ln))
            img.set_data(tempImg)
            vmin=np.min(tempImg)
            vmax=np.max(tempImg)
            img.set_clim(vmin, vmax)
            plt.pause(1e-60)
        
        Time.append(timeB)
        
    return Time,rtImg


def loadData():
    filepath =  r"panel.npy"
    data =np.load(filepath,allow_pickle=True)
    him =np.array( data.item().get('HIM'),"double")
    N=him.shape[0]*him.shape[1] #64*64 = 4096
    #ri=np.reshape(him,(N,him.shape[2])) #降維(3->2) N, 169bands
    him=np.reshape(him,(N,him.shape[2]))
    return him


''''畫圖'''
if __name__ == '__main__':

    startPixel = 170
    MSE_R_RXD=[]
    MSE_K_RXD=[]
    him = loadData() #return reshape過的him

    '''plt RXD'''
    print('R_RXD')
    result_R_RXD = np.reshape(R_RXD_PLOT(him),4096)
    print('K_RXD')
    result_K_RXD = np.reshape(K_RXD_PLOT(him),4096)
    count = 0
    fun = 0
    fun2 = 0
    status = 0
    for i in range(4):
        if(count == 0):
            print('CR_RXD')
            fun = R_RXD
            costTime_R_CASAUL,result_CR_RXD = FUN_RXD_PLOT(him,status,fun)
            count+1
        if(count == 1):
            print('CK_RXD')
            fun = K_RXD
            costTime_K_CASAUL,result_CK_RXD = FUN_RXD_PLOT(him,status,fun)
            count+1
        if(count == 2):
            print('RT_CR_RXD')
            costTime_R_WOODBURY,result_RT__CR_RXD = RT_FUN_RXD_PLOT(him,status,fun,fun2)
            count+1
        if(count == 3):
            print('RT_CK_RXD')
            costTime_K_WOODBURY,result_RT__CK_RXD = RT_FUN_RXD_PLOT(him,status,fun,fun2)
 
    

    '''plt MSE'''
    for i in range(4096):
          MSE_R = mean_squared_error(result_CR_RXD[:i+1],result_RT__CR_RXD[:i+1])
          MSE_R_RXD.append(MSE_R)
          MSE_K = mean_squared_error(result_CK_RXD[:i+1],result_RT__CK_RXD[:i+1])
          MSE_K_RXD.append(MSE_K) 
    plt.figure("MSE_R")
    plt.title("MSE_R")
    plt.plot(range(len(MSE_R_RXD)),MSE_R_RXD)      
    plt.figure("MSE_K")
    plt.title("MSE_K")
    plt.plot(range(len(MSE_K_RXD)),MSE_K_RXD)

    '''plt computing time'''           
    plt.figure("R of computing time")
    plt.title("R of computing time")
    tempA = np.around(np.sum(costTime_R_CASAUL), 5)
    plt.plot(range(len(costTime_R_CASAUL)), costTime_R_CASAUL, 'b',label = 'Casaul : '+ str(tempA), color = 'y')
    tempC = np.around(np.sum(costTime_R_WOODBURY), 5)
    plt.plot(range(len(costTime_R_WOODBURY)), costTime_R_WOODBURY, 'r',label ='WoodBury : '+  str(tempC), color = 'g')
    plt.legend()
    plt.xlim(0,4096)
    plt.figure("K of computing time")
    plt.title("K of computing time")
    tempB = np.around(np.sum(costTime_K_CASAUL), 5)
    plt.plot(range(len(costTime_K_CASAUL)), costTime_K_CASAUL, 'b', label='Casaul : '+ str(tempB), color = 'y')
    tempD = np.around(np.sum(costTime_K_WOODBURY), 5)
    plt.plot(range(len(costTime_K_WOODBURY)), costTime_K_WOODBURY, 'r', label='WoodBury : '+  str(tempD), color = 'g')
    plt.legend()
    plt.xlim(0, 4096)
    
    plt.show()

'''寫入res.npz'''
np.savez('res.npz',
         R_RXD=result_R_RXD,
         K_RXD=result_K_RXD,
         CR_RXD=result_CR_RXD,
         CK_RXD=result_CK_RXD,
         RT_CR_RXD=result_RT__CR_RXD,
         RT_CK_RXD=result_RT__CK_RXD,
         MSE_R=MSE_R_RXD,
         MSE_K=MSE_K_RXD,
         t_cr=costTime_R_CASAUL,
         t_ck=costTime_K_CASAUL,
         t_rt_cr=costTime_R_WOODBURY,
         t_rt_ck=costTime_K_WOODBURY)

