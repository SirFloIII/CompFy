


#theta=(v0,vbar,rho,kap,sigm)
#setwd("C:/Users/thoma/OneDrive/Unizeug/Compfun")
require("NMOF")


Callpreis<-function(theta,S0,mu,K,t){
  tau=t
  r=mu
  q=0
  v0=theta[1]^2
  vT=theta[2]^2
  rho=theta[3]
  k=theta[4]
  sigma=theta[5]
 erg=callHestoncf(S0,K,t,r,q,v0,vT,rho,k,sigma) 
  return(erg)
  
}
theta=c(0.2,0.2,-0.7,3,0.3)
theta2=c(0.1,0.1,-0.8,4,0.2)



GradCall<-function(theta,S0,mu,K,t,eps){
  erg=array(0,5)
  for(i in 1:5){
    v=array(0,5)
    v[i]=eps
    erg[i]=(Callpreis(theta+v,S0,mu,K,t)-Callpreis(theta-v,S0,mu,K,t))/(2*eps)
  }
  return(erg)
  
}

Graddat<-function(theta,S0,mu,eps,daten){
  rr=Resi(theta,S0,mu,daten)
  erg=array(0,5)
  for(i in 1:length(Resi)){
    K=daten[1,i]
    t=daten[2,i]
    erg=erg+GradCall(theta,S0,mu,K,t,eps)*rr[i]
  }
  return(erg)
}

Resi<-function(theta,S0,mu,daten){
  erg=0
  for(i in 1:length(daten[1])){
    erg[i]=(Callpreis(theta,S0,mu,daten[1,i],daten[2,i])-daten[3,i])
  }
  return(erg)
}

GradVerfahren<-function(theta,S0,mu,daten){
  eps=0.0001
  #grad=GradCall(theta,S0,mu,K,t,eps)
  alph=0.1
  tol1=0.00001
  tol2=0.00001
  tol3=0.00001
  
  minerror=10000
  mintheta<-0
  for(i in 1:1000){
    grad=Graddat(theta,S0,mu,eps,daten)
    olderror=sqrt(sum((Resi(theta,S0,mu,daten))^2))
    #grad=grad/sqrt(sum(grad^2))
    #grad=grad/abs(grad)
    grad[2]=grad[2]/sqrt(sum(grad^2))
    grad=grad/sqrt(sum(grad^2))
    thetanew=theta-alph*grad
    
    if(thetanew[2]<0){
      thetanew[2]=0.03
    }
    
    if(thetanew[5]<0){
      thetanew[5]=0.03
    }
    
    if(thetanew[1]>0.6){
      thetanew[1]=0.6
    }
    
    "if(thetanew[3]<-1){
      thetanew[3]=-0.9
    }"
    
    newerror=sqrt(sum((Resi(thetanew,S0,mu,daten))^2))
    
    
    if(newerror<minerror){
      minerror<-newerror
      mintheta<-thetanew
      cat("NeFehl=",newerror,"theta=",thetanew,"\n")
    }
    
    if(newerror<olderror){
      alph=1
      theta=thetanew
    }
    
    if(newerror>olderror){
      alph=alph*0.7
    }
    
    if(sqrt(sum(grad^2))<tol1){
     print("Bed 1")
       return(theta)
    }
    if(newerror<tol2){
      print("Bed2")
      print(olderror)
      print(newerror)
      return(theta)
    }
    if(thetanew[3]<(-1)){
      print("Fuck you")
      return(mintheta)
    }
    
    
  }
  return(mintheta)
  
  
  }

GradVerfahren2<-function(theta,S0,mu,daten){
  eps=0.0001
  #grad=GradCall(theta,S0,mu,K,t,eps)
  alph=0.7
  tol1=0.001
  tol2=0.01
  tol3=0.001
  
  minerror=10000
  mintheta<-0
  for(i in 1:100){
    grad=Graddat(theta,S0,mu,eps,daten)
    olderror=sqrt(sum((Resi(theta,S0,mu,daten))^2))
    grad[2]=0
    grad=grad/sqrt(sum(abs(grad)))
    thetanew=theta-alph*grad
    
    if(thetanew[2]<0){
      thetanew[2]=0.03
    }
    
    newerror=sqrt(sum((Resi(thetanew,S0,mu,daten))^2))
    
    if(newerror<minerror){
      minerror<-newerror
      mintheta<-thetanew
    }
    
    if(newerror<olderror){
      alph=1
      theta=thetanew
    }
    
    if(newerror>olderror){
      alph=alph*0.7
    }
    
    if(sqrt(sum(grad^2))<tol1){
      print("Bed 1")
      return(theta)
    }
    if(newerror<tol2){
      print("Bed2")
      print(olderror)
      print(newerror)
      return(theta)
    }
    if(thetanew[3]<(-1)){
      print("Fuck you")
      return(mintheta)
    }
    return(mintheta)
    
  }
  
  
  
}



GradVerfahren3<-function(theta,S0,mu,daten){
  eps=0.0001
  #grad=GradCall(theta,S0,mu,K,t,eps)
  alph=0.1
  tol1=0.00001
  tol2=0.0001
  tol3=0.00001
  
  minerror=10000
  mintheta<-0
  for(i in 1:1000){
    grad=Graddat(theta,S0,mu,eps,daten)
    olderror=sqrt(sum((Resi(theta,S0,mu,daten))^2))
    #grad=grad/sqrt(sum(grad^2))
    #grad=grad/abs(grad)
    grad[2]=grad[2]/sqrt(sum(grad^2))
    grad=grad/sqrt(sum(grad^2))
    thetanew=theta-alph*grad
    
    thetmin=c(0.03,0.03,-0.9,0.4,0.03)
    thetmax=c(0.95,0.95,-0.05,7,0.95)
    thetanew=pmax(thetanew,thetmin)
    thetanew=pmin(thetanew,thetmax)
    
    "if(thetanew[3]<-1){
    thetanew[3]=-0.9
  }"
    #print(thetanew)
    newerror=sqrt(sum((Resi(thetanew,S0,mu,daten))^2))
    
    
    if(newerror<minerror){
      minerror<-newerror
      mintheta<-thetanew
      cat("NeFehl=",newerror,"theta=",thetanew,"\n")
    }
    
    if(newerror<olderror){
      alph=1
      theta=thetanew
    }
    
    if(newerror>olderror){
      alph=alph*0.7
    }
    
    if(sqrt(sum(grad^2))<tol1){
      print("Bed 1")
      return(theta)
    }
    if(newerror<tol2){
      print("Bed2")
      print(olderror)
      print(newerror)
      return(theta)
    }
    if(thetanew[3]<(-1)){
      print("Fuck you")
      return(mintheta)
    }
    
    
}
  return(mintheta)
  
  
}

Pepsi=read.csv("Pepsi.csv",header = FALSE,dec = ".",sep=";")
Cola=read.csv("Cola.csv",header=FALSE,dec=".",sep=";")

mu=0.005





  #  GradVerfahren(theta,110,0.01,daten)