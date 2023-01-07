import numpy as np

def fRphantom(os,Nse,th,P,doshift):
    s = np.arange(-Nse//2,(Nse//2))/2/os
    A = P[0]
    a = P[1]
    b = P[2]
    x0 = P[3]
    y0 = P[4]
    phi = -P[5]*np.pi/180
    asq = a*a
    bsq = b*b

    Nrho = 4*Nse
    rhosp = np.arange(-Nrho//2,Nrho//2)/Nrho*2
    rho = rhosp.copy()
    theta = th
    Ntheta = len(th)
    rho0 = np.sqrt(asq*(np.cos(theta-phi))**2+bsq*(np.sin(theta-phi))**2)
    g1 = np.zeros([Ntheta,Nse],dtype='complex128')
    rhop = rho.copy()
    ep = np.exp(-2*np.pi*1j*np.outer(rhop,s))
    for it in range(Ntheta):
        # print(x0,y0,theta[it],rho0[it])
        idx = np.abs(rhop+x0*np.cos(theta[it])-y0*np.sin(theta[it]))<=rho0[it]        
        ss1 = 2*A*np.sqrt(asq*bsq)*np.emath.sqrt(rho0[it]**2-(rhop+x0*np.cos(theta[it])-y0*np.sin(theta[it]))**2)/(rho0[it]**2)        
        ss = np.matmul(ep.swapaxes(0,1),(idx*ss1))
        g1[it,:] += ss
    
    g1=g1*(rhosp[1]-rhosp[0])*Nse/2/os
    
    
    if doshift==1:
        g1=g1*np.exp(-2*np.pi*1j*(4*os/Nrho)*s)    
    return g1
        
def wint(n,t):
    N = len(t)
    s = np.linspace(1e-40,1,n,endpoint=True)
    iv = np.linalg.inv(np.exp(np.outer(np.log(s),np.arange(n))))
    v = np.exp(np.outer(np.log(s),np.arange(1,n+2)))/np.arange(1,n+2)
    u = v[1:]-v[:-1]
        
    W1 = np.matmul(u[:,1:n+1],iv)#;%x*pn(x) term
    W2 = np.matmul(u[:,:n],iv)#;%const*pn(x) term
    p = 1/np.concatenate((np.arange(1,n), (n-1)*np.ones(N-2*(n-1)-1), np.arange(1,n)[::-1]))
    
    w = np.zeros(N)
    for j in range(N-n+1):
        W = (t[j+n-1]-t[j])**2*W1+(t[j+n-1]-t[j])*t[j]*W2#%Change coordinates, and constant and linear parts
        for k in range(n-1):#1:n-1,
            w[j:j+n]+=W[k,:]*p[j+k]#;% Add to weights
    w[N-40:N] = (w[N-40]/(N-40))*np.arange(N-40,N)
    return w

def apply_filter_exact(Ntheta,Ns,filter,ellipse):

    os = 4
    d = 1/2
    th = np.linspace(0,np.pi,Ntheta,endpoint=False)
    Nse = os*Ns
    Rhate = np.zeros([Ntheta,Nse])
    for el in ellipse:
        Rhate = Rhate+fRphantom(os,Nse,th,el,1-Ns%2)
    t = np.arange(Nse//2+1)/Nse

    if filter == 'ramp':
        wfa = Nse*0.5*wint(12,t)
    elif filter == 'shepp-logan':
        wfa = Nse*0.5*wint(12,t)*np.sinc(t/(2*d))*(t/d<=2)
    elif filter == 'cosine':
            wfa = Nse*0.5*wint(12,t)*np.cos(pi*t/(2*d))*(t/d<=1)
    elif filter == 'cosine2':
            wfa = Nse*0.5*wint(12,t)*(np.cos(np.pi*t/(2*d)))**2*(t/d<=1)
    elif filter == 'hamming':
            wfa = Nse*0.5*wint(12,t)*(.54 + .46 * np.cos(np.pi*t/d))*(t/d<=1)
    elif filter == 'hann':
            wfa = Nse*0.5*wint(12,t)*(1+np.cos(np.pi*t/d))/2*(t/d<=1)

    wfa = wfa*(wfa>=0)
    wfamid = 2*wfa[0]
    wfae = np.zeros([Nse+1])
    wfae[:Nse//2] = wfa[1:][::-1]
    wfae[Nse//2] = wfamid
    wfae[Nse//2+1:] = wfa[1:]
    wfae = wfae[:-1]
    
    g = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Rhate*wfae)))
    fN = Ns//2
    cN = int(np.ceil(Ns/2))
    g = g[:,Nse//2-fN+1:Nse//2+cN+1].real

    return g