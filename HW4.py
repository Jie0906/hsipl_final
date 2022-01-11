import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import time

start_pixel = 170


def K (him):

    N = him.shape[0]

    ri = np.reshape(him,(N,him.shape[1]))
    u = (np.mean(np.transpose(ri), 1))
    K = np.dot(np.transpose(ri-u),(ri-u))/N

    return u,K



def R (him):
    
    N = him.shape[0]
    ri = np.reshape(him,(N,him.shape[1]))
    R = np.dot(np.transpose(ri),(ri))/N
    
    return R


def Woodbury (invA,U,V):

    VT = V
    V = np.transpose(V)
    U = np.transpose(U)
    invW = invA - (invA@U)@(VT@invA)/(1 + VT@invA@U)
    
    return invW


def k_rxd (him):

    N = him.shape[0]
    try:
        u,Ka = K(him)
        invK = np.linalg.inv(Ka)
    except :
        u,Ka = K(him)
        invK = np.linalg.pinv(Ka)
    r = np.reshape(him,(N,him.shape[1]))
    k_rxd = np.zeros(N)
    for p in range(N):
        ru = (r[p,:]-u)
        k_rxd[p] = ru@invK@np.reshape(ru,(ru.shape[0],1))
    img = k_rxd

    return invK,u,img


def r_rxd(him):

    N = him.shape[0]
    try:
        invR = np.linalg.inv(R(him))
    except :
        invR = np.linalg.pinv(R(him))
    r = np.reshape(him,(N,him.shape[1]))
    r_rxd = np.zeros(N)
    for p in range(N):
        r_rxd[p] = r[p]@invR@np.reshape(r[p],(r[p].shape[0],1))
    img = r_rxd
    return invR,img

def cr_rxd(r,current_img):

    band = r.shape[0]
    try:
        invR = np.linalg.inv(R(current_img))
    except :
        invR = np.linalg.pinv(R(current_img))
    cr_rxd = np.reshape(r,(1,band)) @ invR @ r
    pixel = cr_rxd
    
    return pixel


def ck_rxd(r,current_img):

    band = r.shape[0]
    try:
        u,Ka = K(current_img)
        invK = np.linalg.inv(Ka)
    except :
        u,Ka = K(current_img)
        invK = np.linalg.pinv(Ka)
    ru = r - u
    ck_rxd = np.reshape(ru,(1,band)) @ invK @ ru
    pixel = ck_rxd

    return u,pixel


def rt_cr_rxd(r,n,pre_R):

    band = r.shape[1] 
    rt = np.reshape(r,(band,1))
    parA = ((n-1)/n)*pre_R
    parU = 1/np.sqrt(n)*r
    parV = parU 
    parA = np.linalg.inv(parA)
    invR = Woodbury(parA,parU,parV)
    pixel = r@invR@rt
    return invR,pixel


def rt_ck_rxd(r,n,pre_K,pre_u):
    
    r = np.reshape(r,(1,169))
    u = (1-1/n) * pre_u + (1/n)*r
    band = r.shape[1]
    parA = (1-1/n)*pre_K
    parU = (np.sqrt((n-1))/n)*(pre_u-r) 
    parV =  parU
    parA = np.linalg.inv(parA)
    invK = Woodbury(parA,parU,parV)
    pixel = (r-u) @ invK @ np.reshape(r-u,(band,1))

    return invK,u,pixel

   
def print_r_rxd(him,plot):
    
    invR,img = r_rxd(him)
    rs_img = np.zeros((64, 64))
    rs_img = np.reshape(img,(64,64))
    if plot== True:
        plt.figure()
        plt.title("R-RXD")
        plt.axis("off")
        plt.imshow(rs_img,'gray')
    return img

def print_k_rxd(him,plot):

    invK,u,img = k_rxd(him)
    rs_img = np.zeros((64, 64))
    rs_img = np.reshape(img,(64,64))
    if plot== True:
        plt.figure()
        plt.title("K-RXD")
        plt.axis("off")
        plt.imshow(rs_img,'gray')
    return img

    
def print_cr_rxd(him,plot):
    
    if plot == True:
        plt.figure("cr_rxd")
        plt.title("CR-RXD")
        plt.axis("off")
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.axis("off")
        ax.set_title("CR-RXD")
        img = ax.imshow(np.zeros((64,64)))    
        
    N = him.shape[0]
    l = int(np.sqrt(N))
    rs_img = np.zeros((l,l))
    oneD_img = np.zeros(N)
    time_list = []
    for i in range (N):
                
        if i <= start_pixel :
            s_time = time.perf_counter()
            pre_R,oneD_img[0:i+1] = r_rxd(him[0:i+1])
            e_time = time.perf_counter() - s_time
        if i > start_pixel :
            s_time = time.perf_counter()
            oneD_img[i] = cr_rxd(him[i],him[0:i+1])
            e_time = time.perf_counter() - s_time
        if plot == True:
            rs_img = np.reshape(oneD_img,(l,l))
            img.set_data(rs_img)
            img.set_clim(vmin=np.min(rs_img), vmax=np.max(rs_img))
            plt.pause(1e-60)
            
        time_list.append(e_time)
        
    return time_list,oneD_img

def print_ck_rxd(him,plot):
    
    if plot == True:
        plt.figure("ck_rxd")
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.axis("off")
        ax.set_title("CK-RXD")
        img = ax.imshow(np.zeros((64,64)))
        
    N = him.shape[0]
    l = int(np.sqrt(N))
    rs_img = np.zeros((l,l))
    oneD_img = np.zeros(N)
    time_list = []
    
    for i in range (N):
                
        if i <= start_pixel :
            s_time = time.perf_counter()
            pre_K,u,oneD_img[0:i+1] = k_rxd(him[0:i+1])
            e_time = time.perf_counter() - s_time
        if i > start_pixel :
            s_time = time.perf_counter()
            u,oneD_img[i] = ck_rxd(him[i],him[0:i+1])
            e_time = time.perf_counter() - s_time
        if plot == True:
            rs_img = np.reshape(oneD_img,(l,l))
            img.set_data(rs_img)
            img.set_clim(vmin=np.min(rs_img), vmax=np.max(rs_img))
            plt.pause(1e-60)
            
        time_list.append(e_time)
            
    return time_list,oneD_img

           
    
def print_rt_cr_rxd(him, plot):
    
    if plot == True:
        plt.figure("rt_cr_rxd")
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.axis("off")
        ax.set_title("RT-CR-RXD")
        img = ax.imshow(np.zeros((64,64)))
        
    N = him.shape[0]
    l = int(np.sqrt(N))
    rs_img = np.zeros((l,l))
    oneD_img = np.zeros(N)
    time_list = []
    for i in range (N):
        
        if i <= start_pixel :
            s_time = time.perf_counter()
            pre_R,oneD_img[:i+1]=r_rxd(him[:i+1])
            e_time = time.perf_counter() - s_time
        if i > start_pixel :
            pre_R = np.linalg.inv(pre_R)
            s_time = time.perf_counter()
            pre_R,oneD_img[i:i+1] = rt_cr_rxd(him[i:i+1],i+1 ,pre_R)        
            e_time = time.perf_counter() - s_time
        if plot == True:
            rs_img = np.reshape(oneD_img,(l,l))
            img.set_data(rs_img)
            img.set_clim(vmin=np.min(rs_img), vmax=np.max(rs_img))
            plt.pause(1e-60)
        
        time_list.append(e_time)
        
    return time_list,oneD_img


def print_rt_ck_rxd(him,plot):
    
    plot = False
    if plot == True:
        plt.figure("rt_ck_rxd")
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.axis("off")
        ax.set_title("RT-CK-RXD")
        img = ax.imshow(np.zeros((64,64)))    
        
    N = him.shape[0]
    l = int(np.sqrt(N))
    rs_img = np.zeros((l,l))
    oneD_img = np.zeros(N)
    time_list = []
 
    for i in range (N):
        
        if i <= start_pixel :
            s_time = time.perf_counter()
            pre_K,pre_u,oneD_img[:i+1] = k_rxd(him[:i+1])
            e_time = time.perf_counter() - s_time
        if i > start_pixel :
            pre_K = np.linalg.inv(pre_K)
            s_time = time.perf_counter()            
            pre_K,pre_u,oneD_img[i:i+1] = rt_ck_rxd(him[i:i+1],i+1,pre_K,pre_u)
            e_time = time.perf_counter() - s_time
        if plot == True:
            rs_img = np.reshape(oneD_img,(l,l))
            img.set_data(rs_img)
            img.set_clim(vmin=np.min(rs_img), vmax=np.max(rs_img))
            plt.pause(1e-60)

        time_list.append(e_time)

    return time_list,oneD_img


#[y,x,z]
def init_data():
    filepath =  r"panel.npy"
    data =np.load(filepath,allow_pickle=True)
    him =np.array( data.item().get('HIM'),"double")
    him=np.reshape(him,(4096,169))
    return him

if __name__ == '__main__':
    
    him = init_data()

#   if plot == True then plot figure
    print('R_RXD')
    re1 = np.reshape(print_r_rxd(him,False),4096)
    print('K_RXD')
    re2 = np.reshape(print_k_rxd(him,False),4096)
    print('CK_RXD')
    time4,re4 = print_ck_rxd(him,False)
    print('RT_CK_RXD')
    time6,re6 = print_rt_ck_rxd(him,False)
    print('CR_RXD')
    time3,re3 = print_cr_rxd(him,False)
    print('RT_CR_RXD')
    time5,re5 = print_rt_cr_rxd(him,False)

    mse_r_rxd=[]
    mse_k_rxd=[]
    
    for i in range(4096):
          mse_r = mean_squared_error(re3[:i+1],re5[:i+1])
          mse_r_rxd.append(mse_r)
          mse_k = mean_squared_error(re4[:i+1],re6[:i+1])
          mse_k_rxd.append(mse_k)
          
    plt.figure("mse_R")
    plt.title("MSE_R")
    plt.plot(range(len(mse_r_rxd)),mse_r_rxd)
             
    plt.figure("mse_K")
    plt.title("MSE_K")
    plt.plot(range(len(mse_k_rxd)),mse_k_rxd)
             
    plt.figure("computing time R")
    plt.title("computing time R")
    t3 = np.around(np.sum(time3), 5)
    plt.plot(range(len(time3)), time3, 'b',label = 'Casaul : '+ str(t3), color = 'r')
    t5 = np.around(np.sum(time5), 5)
    plt.plot(range(len(time5)), time5, 'r',label ='Woodbury : '+  str(t5), color = 'b')
    plt.legend()
    plt.xlim(0,4096)
    
    plt.figure("computing time K")
    plt.title("computing time K")
    t4 = np.around(np.sum(time4), 5)
    plt.plot(range(len(time4)), time4, 'b', label='Casaul : '+ str(t4), color = 'r')
    t6 = np.around(np.sum(time6), 5)
    plt.plot(range(len(time6)), time6, 'r', label='Woodbury : '+  str(t6), color = 'g')
    plt.legend()
    plt.xlim(0, 4096)
    
    plt.show()

# npz
np.savez('res.npz',
         r_rxd=re1,k_rxd=re2,
         cr_rxd=re3,ck_rxd=re4,
         rt_cr_rxd=re5,rt_ck_rxd=re6,
         mse_r=mse_r_rxd,
         mse_k=mse_k_rxd,
         t_cr=time3,
         t_ck=time4,
         t_rt_cr=time5,
         t_rt_ck=time6)

