import numpy as np

from scipy.special import jv

def fphantom(os,Ne,P,doshift):
    A = P[0]
    a = P[1]
    b = P[2]
    x0 = P[3]
    y0 = P[4]
    phi = P[5]*np.pi/180
    asq = a*a
    bsq = b*b

    [k1,k2] = np.meshgrid(np.arange(-Ne//2,Ne//2+1)/2/os,np.arange(-Ne//2,Ne//2+1)/2/os)
    k1h = np.cos(phi)*k1-np.sin(phi)*k2
    k2h = np.sin(phi)*k1+np.cos(phi)*k2
    K = np.sqrt((a*k1h)**2+(b*k2h)**2)
    be = jv(1,2*np.pi*K)
    with np.errstate(divide='ignore', invalid='ignore'):            
        fhate1 = np.divide(A*a*b*np.pi*be,np.pi*K)*np.exp(-2*np.pi*1j*(k1*(-x0)+k2*(y0)))
    fhate1[Ne//2,Ne//2] = A*a*b*np.pi*1#;%(1-(pi*K).^2+(pi*K).^4/12);%approx Taylor
    fhate1 *= (Ne/os/2)**2
    if doshift==1:
        fhate1 = fhate1*np.exp(-2*np.pi*1j*(1*os/Ne*k1+1*os/Ne*k2))
    fhate1 = fhate1[:-1,:-1]
    return fhate1


def apply_filter_2d_exact(f,filter,ellipse):
    os = 4
    d = 1/2
    N = f.shape[1]
    Ne = os*N
    [t1,t2] = np.meshgrid(np.arange(Ne/2+1)/Ne,np.arange(Ne/2+1)/Ne)
    t = np.sqrt(t1**2+t2**2)

    if filter == 'ramp':
        return f

    fhate = np.zeros([Ne,Ne])
    for el in ellipse:
        fhate = fhate+fphantom(os,Ne,el,1-N%2)

    if filter=='ramp':
        pass
    elif filter == 'shepp-logan':
        wfa = np.sinc(t/(2*d))*(t/d<=2*np.sqrt(2))
    elif filter == 'cosine':
        wfa = np.cos(np.pi*t/(2*d))*(t/d<=1)
    elif filter == 'cosine2':
        wfa = (np.cos(np.pi*t/(2*d)))**2*(t/d<=1)
    elif filter == 'hamming':
        wfa = (.54 + .46 * np.cos(np.pi*t/d))*(t/d<=1)
    elif filter == 'hann':
        wfa = (1+np.cos(np.pi*t/d))/2*(t/d<=1)
    else:
        print('filter is not implemented')
        exit(1)

    # fill array for 2D filter 
    wfa = wfa*(wfa>=0)    
    wfae = np.zeros([Ne+1,Ne+1])
    wfae[:Ne//2+1,:Ne//2+1] = wfa[::-1,::-1]
    wfae[:Ne//2+1,Ne//2:] = wfa[::-1,:]
    wfae[Ne//2:,:Ne//2+1] = wfa[:,::-1]
    wfae[Ne//2:,Ne//2:] = wfa
    wfae = wfae[:-1,:-1]
    g = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(wfae*fhate))).real
    fN = N//2
    cN = int(np.ceil(N/2))
    g = g[Ne//2-fN:Ne//2+cN,Ne//2-fN:Ne//2+cN]
    
    return g

