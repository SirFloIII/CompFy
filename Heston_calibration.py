# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:46:20 2019

@author: Thomas
"""
#Annahme: data sind die Daten als nx3 Matrix mit Spalten: K|T|Callpreis
#theta=(v0,vbar,rho,kap,sigm)

import numpy as np
import cmath as cm


def xi(theta,u):
    return theta[3]-theta[4]*theta[2]*u*(1j)


def de(theta,u):
    return cm.sqrt(xi(theta,u)**2+(theta[4]**2)*(u**2+1j*u))

def g1(theta,u):
    return (xi(theta,u)+de(theta,u))/(xi(theta,u)-de(theta,u))

def g2(theta,u):
    return 1/g1(theta,u)

def A1(theta,u,t):
    return (u**2+1j*u)*np.sinh(de(theta,u)*t/2)

def A2(theta,u,t):
    return (de(theta,u)*np.cosh(de(theta,u)*t/2))+xi(theta,u)*np.sinh(de(theta,u)*t/2)

def AA(theta,u,t):
    return A1(theta,u,t)/A2(theta,u,t)

def BB(theta,u,t):
    return de(theta,u)*cm.exp(theta[3]*t/2)/A2(theta,u,t)

"""def phi(theta,u,t,S0,mu):
    a=1j*u*(cm.log(S0)+mu*t)
    #r=mu????
    b=theta[3]*theta[1]/(theta[4]**2)*((xi(theta,u)+de(theta,u))*t-2*cm.log((1-g1(theta,u)*cm.exp(de(theta,u)*t))/(1-g1(theta,u))))
    c=theta[0]/(theta[4]**2)*(xi(theta,u)+de(theta,u))*(1-cm.exp(de(theta,u)*t))/(1-g1(theta,u)*cm.exp(de(theta,u)*t))
    return cm.exp(a+b+c)"""
    #theta=(v0,vbar,rho,kap,sigm)
def phi(theta,u,t,S0,mu):
    a=1j*u*(cm.log(S0)+mu*t)-t*theta[3]*theta[1]*theta[2]*1j*u/(theta[4])-theta[0]*AA(theta,u,t)
    b=2*theta[3]*theta[1]/(theta[4]**2)*DD(theta,u,t)
    return cm.exp(a+b)

def phi2(theta,u,t,S0,mu):
    a=1j*u*(cm.log(S0)+mu*t)
    #r=mu????
    b=theta[3]*theta[1]/(theta[4]**2)*((xi(theta,u)-de(theta,u))*t-2*cm.log((1-g2(theta,u)*cm.exp(-de(theta,u)*t))/(1-g2(theta,u))))
    c=theta[0]/(theta[4]**2)*(xi(theta,u)-de(theta,u))*(1-cm.exp(-de(theta,u)*t))/(1-g2(theta,u)*cm.exp(-de(theta,u)*t))
    return cm.exp(a+b+c)




"""def DD(theta,u,t):
    return cm.log(BB(theta,u,t))"""

def DD(theta,u,t):
    return cm.log(de(theta,u))+(theta[3]-de(theta,u))*t/2-cm.log((de(theta,u)+xi(theta,u))/2+(de(theta,u)-xi(theta,u))/2*cm.exp(-de(theta,u)*t))

def dddrho(theta,u):
    return -xi(theta,u)*theta[4]*1j*u/de(theta,u)

def dA2drho(theta,u,t):
    return -theta[4]*1j*u*(2+t*xi(theta,u))/(2*de(theta,u))*(xi(theta,u)*cm.cosh(de(theta,u)*t/2)+de(theta,u)*cm.sinh(de(theta,u)*t/2))

def dBdrho(theta,u,t):
    return cm.exp(theta[3]*t/2)*(1/A2(theta,u,t)*dddrho(theta,u)-de(theta,u)/(A2(theta,u,t)**2)*dA2drho(theta,u,t))

def dA1drho(theta,u,t):
    return -1j*u*(u**2+1j*u)*t*xi(theta,u)*theta[4]/(2*de(theta,u))*cm.cosh(de(theta,u)*t/2)

def dAdrho(theta,u,t):
    return 1/A2(theta,u,t)*dA1drho(theta,u,t)-AA(theta,u,t)/A2(theta,u,t)*dA2drho(theta,u,t)

def dBdkap(theta,u,t):
    return 1j/(theta[4]*u)*dBdrho(theta,u,t)+t*BB(theta,u,t)/2


def dddsigm(theta,u):
    return (theta[2]/theta[4]-1/xi(theta,u))*dddrho(theta,u)+theta[4]/de(theta,u)*(u**2)
    
def dA1dsigm(theta,u,t):
    return (u**2*1j*u)*t/2*dddsigm(theta,u)*cm.cosh(de(theta,u)*t/2)   

def dA2dsigm(theta,u,t):
    a=theta[2]/theta[4]*dA2drho(theta,u,t)
    b=-(2+t*xi(theta,u))/(1j*u*t*xi(theta,u))*dA1drho(theta,u,t)
    c=theta[4]*t*A1(theta,u,t)/2
    return a+b+c

def dAdsigm(theta,u,t):
    return 1/A2(theta,u,t)*dA1dsigm(theta,u,t)-AA(theta,u,t)/A2(theta,u,t)*dA2drho(theta,u,t)

def h1(theta,u,t):
    return -AA(theta,u,t)

def h2(theta,u,t):
    return 2*theta[3]/(theta[4]**2)*DD(theta,u,t)-t*theta[3]*theta[2]*1j*u/theta[4]

def h3(theta,u,t):
    return -theta[0]*dAdrho(theta,u,t)+2*theta[3]*theta[1]/((theta[4]**2)*de(theta,u))*(dddrho(theta,u)-de(theta,u)/A2(theta,u,t)*dA2drho(theta,u,t))-t*theta[3]*theta[1]*1j*u/theta[4]

def h4(theta,u,t):
    a=theta[0]/(theta[4]*1j*u)*dAdrho(theta,u,t)+2*theta[1]/(theta[4]**2)*DD(theta,u,t)
    b=2*theta[3]*theta[1]/(BB(theta,u,t)*theta[4]**2)*dBdkap(theta,u,t)
    c=-t*theta[1]*theta[2]*1j*u/theta[4]
    return a+b+c

def h5(theta,u,t):
    a=-theta[0]*dAdrho(theta,u,t)-4*theta[3]*theta[1]/(theta[4]**3)*DD(theta,u,t)
    b=2*theta[3]*theta[1]/(de(theta,u)*theta[4]**2)*(dddsigm(theta,u)-de(theta,u)/A2(theta,u,t)*dA2dsigm(theta,u,t))
    c=t*theta[3]*theta[1]*theta[2]*1j*u/(theta[4]**2)
    return a+b+c

def uu(k,N,Nmax):
    return k/N*Nmax

def dphiint1(theta,t,S0,mu,K,N,Nmax):
    #ACHTUNG: Bei u=0 gibt es eventuell Probleme, deshalb hab ich es einfach ignoriert
    
    h=[[0],[0],[0],[0],[0]]
    """for i in range(5):
        h[i].clear()"""
        
    for k in range(1,N):
        h[0].append(h1(theta,uu(k,N,Nmax)-1j,t))
        h[1].append(h2(theta,uu(k,N,Nmax)-1j,t))
        h[2].append(h3(theta,uu(k,N,Nmax)-1j,t))
        h[3].append(h4(theta,uu(k,N,Nmax)-1j,t))
        h[4].append(h5(theta,uu(k,N,Nmax)-1j,t))
        
    erg=[0,0,0,0,0]
    #ist Anfangswert der numerischen Integralberechnung, könnte man sich ausrechnen, ist aber mühsam, deshalb weggelassen
    for i in range(5):    
        for k in range(1,N):
        #Trapezregel
        #ein bissl aufpassen bei der 0
            erg[i]=erg[i]+K**(-1j*uu(k,N,Nmax))/(1j*uu(k,N,Nmax))*phi(theta,uu(k,N,Nmax)-1j,t,S0,mu)*h[i][k]*(uu(N,N,Nmax)-uu(1,N,Nmax))/N
    
    return np.array(erg).real




#Unterschied der Beiden: einmal hat man phi(u-i) und einmal phi(u)

def dphiint2(theta,t,S0,mu,K,N,Nmax):
    #ACHTUNG: Bei u=0 gibt es eventuell Probleme, deshalb hab ich es einfach ignoriert
    
    h=[[0],[0],[0],[0],[0]]
    """for i in range(5):
        h[i].clear()"""
        
    for k in range(1,N):
        h[0].append(h1(theta,uu(k,N,Nmax),t))
        h[1].append(h2(theta,uu(k,N,Nmax),t))
        h[2].append(h3(theta,uu(k,N,Nmax),t))
        h[3].append(h4(theta,uu(k,N,Nmax),t))
        h[4].append(h5(theta,uu(k,N,Nmax),t))
        
    erg=[0,0,0,0,0]
    #ist Anfangswert der numerischen Integralberechnung, könnte man sich ausrechnen, ist aber mühsam, deshalb weggelassen
    for i in range(5):    
        for k in range(1,N):
        #Trapezregel
        #ein bissl aufpassen bei der 0
            erg[i]=erg[i]+K**(-1j*uu(k,N,Nmax))/(1j*uu(k,N,Nmax))*phi(theta,uu(k,N,Nmax),t,S0,mu)*h[i][k]*(uu(N,N,Nmax)-uu(1,N,Nmax))/N
    
    return np.array(erg).real


"""def dphiint3(theta,t,S0,mu,K,N,Nmax):
    #dient nur zu Testzwecken
    h=[[0],[0],[0],[0],[0]]
    
        
    for k in range(1,N):
        h[0].append(h1(theta,uu(k,N,Nmax),t))
        h[1].append(h2(theta,uu(k,N,Nmax),t))
        h[2].append(h3(theta,uu(k,N,Nmax),t))
        h[3].append(h4(theta,uu(k,N,Nmax),t))
        h[4].append(h5(theta,uu(k,N,Nmax),t))
   
    
    erg=[0]
    #ist Anfangswert der numerischen Integralberechnung, könnte man sich ausrechnen, ist aber mühsam, deshalb weggelassen
    i=0
    for k in range(1,N):
        #Trapezregel
        #ein bissl aufpassen bei der 0
        erg[i]=erg[i]+K**(-1j*uu(k,N,Nmax))/(1j*uu(k,N,Nmax))*phi(theta,uu(k,N,Nmax),t,S0,mu)*h[4][k]*(uu(N,N,Nmax)-uu(1,N,Nmax))/N
    
    return np.array(erg).real"""








def gradC(theta,t,S0,mu,K,N=8593,Nmax=100):
    return cm.exp(-mu*t)/np.pi*(dphiint1(theta,t,S0,mu,K,N,Nmax)-K*dphiint2(theta,t,S0,mu,K,N,Nmax))



def phiint1(theta,t,S0,mu,K,N,Nmax):
    erg=0
    for k in range(1,N):
        erg=erg+K**(-1j*uu(k,N,Nmax))/(1j*uu(k,N,Nmax))*phi(theta,uu(k,N,Nmax)-1j,t,S0,mu)*(uu(N,N,Nmax)/N)
    
    return erg.real
#Unterschied der Beiden: einmal hat man phi(u-i) und einmal phi(u)

def phiint2(theta,t,S0,mu,K,N,Nmax):
    erg=0
    for k in range(1,N):
        erg=erg+K**(-1j*uu(k,N,Nmax))/(1j*uu(k,N,Nmax))*phi(theta,uu(k,N,Nmax),t,S0,mu)*(uu(N,N,Nmax)/N)
    
    return erg.real



#Annahme: data sind die Daten als nx3 Matrix mit Spalten: K|T|Callpreis

def rr(theta,S0,mu,data,N=8593,Nmax=100):
    r=[]
    #Es sollte gelten, dass len(data)=n, sonst bitte code ändern
    for i in range(len(data)):
        r.append(CallPreisH(theta,S0,mu,data[i][0],data[i][1],N,Nmax)-data[i][2])
    
    return np.array(r)
        
def f(theta,S0,mu,data,N=8593,Nmax=100):
    return 1/2*np.dot(rr(theta,S0,mu,data,N,Nmax),rr(theta,S0,mu,data,N,Nmax))


def CallPreisH(theta,S0,mu,K,T,N,Nmax):
    a=1/2*(S0-cm.exp(-mu*T)*K)
    b=cm.exp(-mu*T)/np.pi*phiint1(theta,T,S0,mu,K,N,Nmax)
    c=-cm.exp(-mu*T)/np.pi*K*phiint2(theta,T,S0,mu,K,N,Nmax)
    return a+b+c

#Annahme: data sind die Daten als nx3 Matrix mit Spalten: K|T|Callpreis

def JJ(theta,S0,mu,data,N=8593,Nmax=100):
    J=[]
    for i in range(len(data)):
        J.append(gradC(theta,data[i][1],S0,mu,data[i][0],N,Nmax))
    
    J=np.array(J)
    J=np.transpose(J)
    return J

def LevMarquCali(theta,S0,mu,data,N=8593,Nmax=100):
    eps1=5
    eps2=2
    eps3=0.0001
    #r=r(theta_k)
    #rnew=r(theta_{k+1})
    #genauso thetanew
    r=rr(theta,S0,mu,data,N,Nmax)
    if np.linalg.norm(r)<eps1:
        return theta
    
    v=2
    tau=0.2
    J=JJ(theta,S0,mu,data,N,Nmax)
    #damp=tau*np.max(np.diag(J))
    damp=1
    if damp<=0:
        print("Es gibt wahrscheinlich ein Problem mit mu0")
    
    for k in range(20):
        print(theta)
        df=np.dot(J,r)
        A=np.linalg.inv(J@np.transpose(J)+damp*np.eye(5))
        deltathet=np.dot(A,df)
        
        thetanew=theta+0.1*deltathet
        #ich hab hier den Faktor 0.1 dazugegeben, der hat Wunder bewirkt
        rnew=rr(thetanew,S0,mu,data,N,Nmax)
        
        a=np.transpose(deltathet)
        b=damp*deltathet
        c=np.dot(J,r)
        deltaL=np.dot(a,b+c)
        
        deltaF=np.linalg.norm(r)-np.linalg.norm(rnew)
        
        #print("deltaL=",deltaL,"  deltaF=",deltaF)
        
        if deltaL>0:
             #and deltaF>0
            if (k%3==0) or deltaF>0:
                Jnew=JJ(thetanew,S0,mu,data,N,Nmax)
                J=Jnew
                print("Jnew")
            else:
                damp=damp*v
                v=2*v
        else:
            damp=damp*v
            v=2*v
            
        if np.linalg.norm(r)<eps1:
            print("Cond1")
            return [theta,r]
        
        if max(abs(np.dot(J,[1]*len(J[1]))))<eps2:
            print("Cond2")
            return [theta,r]
        
        if np.linalg.norm(deltathet)/np.linalg.norm(theta)<eps3:
            print("Cond3")
            return [theta,r]
        
        theta=thetanew
        r=rnew
        
    return [theta,r]





theta=[0.08,0.1,-0.8,3,0.25]


S0=110
mu=0.01
N=70
Nmax=5



#theta=(v0,vbar,rho,kap,sigm)









