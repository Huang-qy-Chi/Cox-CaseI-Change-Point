#setwd("C:\\Users\\txw\\Desktop\\change point\\My Programme")
library(optimx)
library(pracma)
library(rootSolve)
library(MASS)
#source("bploy.R")

#baseline hazard: m=3
Lambda0 = function(Time,phi){ 
  n = length(Z2)
  m = length(phi)-1
  u = max(Time) #
  v = min(Time)
  phi_s = cumsum(exp(phi))
  B = c()
  for (s in 0:m){ #Berstein Poly
    B = cbind(B,sapply(Time, bpoly,u=u,v=v,m=m,k=s))
  }   
  return (B%*%matrix(phi_s,ncol=1))
}

#log-likelihood
ln = function(phi, Time, Z1, Z2, theta, eta){ 
  n = length(Z2)
  m = length(phi)-1
  Z = cbind(Z1[,1],Z1[,1]*(Z2>eta),rep(1,n)*(Z2>eta))
  h1 = as.matrix(exp(Z%*%theta)) #Z*theta
  B = c()
  for(s in 0:m){  #Berstein polynomials matrix
    B = cbind(B,sapply(Time, bpoly,u=max(Time),v=min(Time),m=m,k=s))
  }
  phi_s <- cumsum(exp(phi))
  A1 = as.matrix(B%*%phi_s)    #Lambda0n
  B1 = t(A1)%*%(diag(-D))%*%h1 #Censor
  G = function(u){
    return(log(1-exp(-u)))
  }
  sapply(A1*h1 ,G)
  B2 = (1-D)%*%sapply(A1*h1, G)  #Not Censor
  return(B1+B2)
}


estNMLE = function(Z1, Z2, Time, D, m = 3, graph = F, err = 1e-5){
  n = length(Z2)
  d = length(Z1[1,])
  p = (d-1)/2
  theta0 = c(-0.5,rep(1,d-1))  #initial value of theta
  eta0 = mean(Z2)   #initial value of eta
  u = as.numeric(min(Time))
  v = as.numeric(max(Time))
  interval_eta = c(mean(Z2)-0.5, mean(Z2)+0.5)
  grid_eta = seq(interval_eta[1],interval_eta[2],by=0.01)
  index = 0
  #iterating profile estimation
  for(r in 1:100){
    #step 1: estimate Lambda0: Nelder-Mead Method
    Z = cbind(Z1[,1:p],Z1[,(p+1):(2*p)]*(Z2>eta0),rep(1,n)*(Z2>eta0))
    #Berstein Polynominal
    B = c()
    for (s in 0:m){ 
      B = cbind(B,sapply(Time, bpoly,u = max(Time), 
                         v = min(Time),m = m, k = s))
    } 
    #log-likelihood w.r.t Lambda
    #grad
    #gradient(f=l_Lambda,x=c(phi),Time=Time, Z1=Z1, Z2=Z2, theta=theta, eta=eta)
    phi = optim(rep(0,m+1), ln, Z1 = Z1 , Z2 = Z2,
                Time = Time, theta = theta0, eta = eta0,
                control = c(fnscale=-1))$par
    #Lambda0(Time,phi)  #value of baseline hazard
    #plot(sort(Time),sort(Lambda0(Time,phi)),type = "b")  #graph of Lambda0
    #lines(sort(Time),sort(Time))
    Lam = Lambda0(Time, phi)
    
    #step 2: estimate theta: Nelder-Mead Method
    theta1 = theta0
    theta0 = optim(theta1, ln, Z1 = Z1, Z2 = Z2, Time = Time, 
                   control = c(fnscale=-1), phi = phi, eta = eta0)$par
    
    
    #step 3: estimate eta, Brent Method
    eta1 = eta0
    #eta0 = optim(eta1, ln, Z1 = Z1, Z2 = Z2, Time = Time, phi = phi,
                #theta = theta0, control = c(fnscale=-1),
                #lower = interval_eta[1], upper = interval_eta[2],
                #method = "Brent")$par
    
    efind = which.max(sapply(grid_eta, ln, theta = theta0, Time = Time, Z1 = Z1,
           Z2 = Z2, phi = phi))
    eta0 = grid_eta[efind]
    
    cont = c(theta0,eta0)-c(theta1,eta1)
    if(t(cont)%*%cont<err){
      index = 1
      break
    }
  }
  theta = theta0
  eta = eta0
  phi = optim(rep(0,m+1), ln, Z1 = Z1 , Z2 = Z2,
              Time = Time, theta = theta, eta = eta,
              control = c(fnscale=-1))$par
  V1 = sort(Lambda0(Time,phi))
  #V2 = sort(Time)
  if(graph==T){
    plot(sort(Time),V1,type = "b")  #graph of Lambda0
    #lines(sort(Time),V2)
  }
  return(list(phi = phi, theta = theta, eta = eta, index = index))
}
###################################################################################
#multivariate response Z1
lnm = function(phi, Time, Z1, Z2, theta, eta, D){ 
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  m = length(phi)-1
  Z = cbind(Z1,Z1*(Z2>eta),rep(1,n)*(Z2>eta))
  h1 = as.matrix(exp(Z%*%theta)) #Z*theta
  B = c()
  for(s in 0:m){  #Berstein polynomials matrix
    B = cbind(B,sapply(Time, bpoly,u=max(Time),v=min(Time),m=m,k=s))
  }
  phi_s <- cumsum(exp(phi))
  A1 = as.matrix(B%*%phi_s)    #Lambda0n
  B1 = t(A1)%*%(diag(-D))%*%h1 #Censor
  G = function(u){
    return(log(1-exp(-u)))
  }
  #sapply(A1*h1 ,G)
  A2 = sapply(A1*h1, G)
  #A2 = replace(A2, A2 <1e-4 , 1e-4)
  #summary(A2)
  B2 = (1-D)%*%A2  #Not Censor
  return(B1+B2)
}

estNMLE_m = function(Z1, Z2, Time, D, m = 3, graph = F, err = 1e-5,M = 50){
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  #theta0 = c(-0.5,rep(1,d-1))  #initial value of theta
  theta0 = rep(0,d)
  eta0 = mean(Z2)   #initial value of eta
  #phi0 = rep(0,m+1)
  u = as.numeric(min(Time))
  v = as.numeric(max(Time))
  interval_eta = c(mean(Z2)-0.5, mean(Z2)+0.5)
  grid_eta = seq(interval_eta[1],interval_eta[2],by=0.01)
  index = 0
  #iterating profile estimation
  Z = cbind(Z1,Z1*(Z2>eta0),rep(1,n)*(Z2>eta0))
  #Berstein Polynominal
  B = c()
  for (s in 0:m){ 
    B = cbind(B,sapply(Time, bpoly,u = max(Time), 
                       v = min(Time),m = m, k = s))
  } 
  for(r in 1:M){
    #step 1: estimate Lambda0: Nelder-Mead Method
    
    #log-likelihood w.r.t Lambda
    #grad
    #gradient(f=l_Lambda,x=c(phi),Time=Time, Z1=Z1, Z2=Z2, theta=theta, eta=eta)
    #phi = optim(rep(0,m+1), lnm, Z1 = Z1 , Z2 = Z2,
                #Time = Time, theta = theta0, eta = eta0,
                #control = c(fnscale=-1))$par
    #phi1 = phi0
    phi = optim(rep(0,m+1), lnm, Z1 = Z1 , Z2 = Z2, D = D,
                Time = Time, theta = theta0, eta = eta0,
                control = c(fnscale=-1))$par
    #Lambda0(Time,phi)  #value of baseline hazard
    #plot(sort(Time),sort(Lambda0(Time,phi)),type = "b")  #graph of Lambda0
    #lines(sort(Time),sort(Time))
    Lam = Lambda0(Time, phi)
    
    #step 2: estimate theta: Nelder-Mead Method
    theta1 = theta0
    theta0 = optim(theta1, lnm, Z1 = Z1, Z2 = Z2, Time = Time, D = D,
                   control = c(fnscale=-1), phi = phi, eta = eta0)$par
    
    
    #step 3: estimate eta, Brent Method
    eta1 = eta0
    #eta0 = optim(eta1, ln, Z1 = Z1, Z2 = Z2, Time = Time, phi = phi,
    #theta = theta0, control = c(fnscale=-1),
    #lower = interval_eta[1], upper = interval_eta[2],
    #method = "Brent")$par
    
    efind = which.max(sapply(grid_eta, lnm, theta = theta0, Time = Time, Z1 = Z1,
                             Z2 = Z2, phi = phi, D = D))
    eta0 = grid_eta[efind]
    
    cont = c(theta0,eta0)-c(theta1,eta1)
    if(t(cont)%*%cont<err){
      index = 1
      break
    }
  }
  theta = theta0
  eta = eta0
  phi = optim(rep(0,m+1), lnm, Z1 = Z1 , Z2 = Z2, D = D,
              Time = Time, theta = theta, eta = eta,
              control = c(fnscale=-1))$par
  V1 = sort(Lambda0(Time,phi))
  #V2 = sort(Time)
  if(graph==T){
    plot(sort(Time),V1,type = "b")  #graph of Lambda0
    #lines(sort(Time),V2)
  }
  return(list(phi = phi, theta = theta, eta = eta, index = index))
}


#Newton-Raphson method for theta:small grid of eta
estNMLE_nr = function(Z1, Z2, Time, D, m = 3, graph = F, err = 1e-5,M = 50){
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  theta0 = rep(0,d)  #initial value of theta
  eta0 = mean(Z2)   #initial value of eta
  #phi0 = rep(0,m+1)
  u = as.numeric(min(Time))
  v = as.numeric(max(Time))
  interval_eta = c(mean(Z2)-0.5, mean(Z2)+0.5)
  grid_eta = seq(interval_eta[1],interval_eta[2],by=0.01)
  index = 0
  #iterating profile estimation
  Z = cbind(Z1,Z1*(Z2>eta0),rep(1,n)*(Z2>eta0))
  #Berstein Polynominal
  B = c()
  for (s in 0:m){ 
    B = cbind(B,sapply(Time, bpoly,u = max(Time), 
                       v = min(Time),m = m, k = s))
  } 
  for(r in 1:M){
    #step 1: estimate Lambda0: Nelder-Mead Method with Berstein Polynominal
    phi = optim(rep(0,m+1), lnm, Z1 = Z1 , Z2 = Z2, D = D,
                Time = Time, theta = theta0, eta = eta0,
                control = c(fnscale=-1))$par   #parameter of baseline
    Lam = Lambda0(Time, phi)    #value of baseline hazard 
    
    #step 2: estimate theta: Newton-Raphson Method
    theta1 = theta0
    r0 = Lam*exp(Z%*%theta1)  #culmulative hazard function
    r1 = exp(-r0) #survival function
    score1 = -D*r0+(1-D)*(r1*r0)/(1-r1)
    score = t(t(score1)%*%Z)   #score function
    cong = score1^2
    info =  t(Z)%*%diag(as.vector(cong))%*%Z   #information
    invinfo = solve(info)
    theta0 = theta1 + invinfo%*%score #Newton-Raphson
    
    
    #step 3: estimate eta, grid search
    eta1 = eta0
    efind = which.max(sapply(grid_eta, lnm, theta = theta0, Time = Time, Z1 = Z1,
                             Z2 = Z2, phi = phi, D = D))
    eta0 = grid_eta[efind]   #grid search 
    
    cont = c(theta0,eta0)-c(theta1,eta1)
    if(t(cont)%*%cont<err){   #whether convergece
      index = 1   #convergence
      break
    }
  }
  theta = as.vector(theta0)
  eta = eta0
  phi = optim(rep(0,m+1), lnm, Z1 = Z1 , Z2 = Z2, D = D,
              Time = Time, theta = theta, eta = eta,
              control = c(fnscale=-1))$par    #renew the baseline
  V1 = sort(Lambda0(Time,phi))
  var.theta = diag(solve(info/n))/n   
  sd.theta = sqrt(var.theta)            #the standard error
  z.score = theta*(var.theta)^(-1)*(theta)  #significance of one parameter
  p.va = sapply(z.score, pchisq, df = 1)
  p.value = 1-p.va   #significance
  #V2 = sort(Time)
  if(graph==T){
    plot(sort(Time),V1,type = "b")  #graph of baseline Lambda0
    #lines(sort(Time),V2)
  }
  return(list(phi = phi, theta = theta, sd.theta = sd.theta, 
              p.value = p.value, eta = eta, index = index))
}


##########################################################################################

#For a single Cox model without change point
ln_cox = function(phi, Time, D, Z1, theta){ 
  n = length(Z2)
  p = ncol(Z1)
  m = length(phi)-1
  #Z = cbind(Z1[,1:p],Z1[,1:p]*(Z2>eta),rep(1,n)*(Z2>eta))
  h1 = as.matrix(exp(as.matrix(Z1%*%theta))) #Z*theta
  B = c()
  for(s in 0:m){  #Berstein polynomials matrix
    B = cbind(B,sapply(Time, bpoly,u=max(Time),v=min(Time),m=m,k=s))
  }
  phi_s <- cumsum(exp(phi))
  A1 = as.matrix(B%*%phi_s)    #Lambda0n
  B1 = t(A1)%*%(diag(-D))%*%h1 #Censor
  G = function(u){
    return(log(1-exp(-u)))
  }
  #sapply(A1*h1 ,G)
  B2 = (1-D)%*%sapply(A1*h1, G)  #Not Censor
  return(B1+B2)
}

estNMLE_cox = function(Z1, Time, D, m = 3, graph = F, err = 1e-5,M = 50){
  n = nrow(Z1)
  p = ncol(Z1)
  theta0 = rep(0, p)  #initial value of theta
                               #initial value of eta
  u = as.numeric(min(Time))
  v = as.numeric(max(Time))
  index = 0
  #Berstein Polynominal
  B = c()
  for (s in 0:m){ 
    B = cbind(B,sapply(Time, bpoly,u = max(Time), 
                       v = min(Time),m = m, k = s))
  } 
  #iterating profile estimation
  for(r in 1:M){
    #step 1: estimate Lambda0: Nelder-Mead Method
    #Z = cbind(Z1[,1:p],Z1[,(p+1):(2*p)]*(Z2>eta0),rep(1,n)*(Z2>eta0))
    
    #log-likelihood w.r.t Lambda
    #grad
    #gradient(f=l_Lambda,x=c(phi),Time=Time, Z1=Z1, Z2=Z2, theta=theta, eta=eta)
    phi = optim(rep(0,m+1), ln_cox, Z1 = Z1, D = D,
                Time = Time, theta = theta0, 
                control = c(fnscale=-1))$par
    #Lambda0(Time,phi)  #value of baseline hazard
    #plot(sort(Time),sort(Lambda0(Time,phi)),type = "b")  #graph of Lambda0
    #lines(sort(Time),sort(Time))
    Lam = Lambda0(Time, phi)
    
    #step 2: estimate theta: Newton-Raphson Method
    theta1 = theta0
    #theta0 = optim(theta1, ln_cox, Z1 = Z1, Time = Time, 
                   #control = c(fnscale=-1), phi = phi, D = D)$par
    r0 = Lam*exp(Z1%*%theta1)  #culmulative hazard function
    r1 = exp(-r0) #survival function
    score1 = -D*r0+(1-D)*(r1*r0)/(1-r1)
    score = t(t(score1)%*%Z1)   #score function
    cong = score1^2
    info =  t(Z1)%*%diag(as.vector(cong))%*%Z1   #information
    theta0 = theta1 + ginv(info)%*%score
    
    cont = theta0 - theta1
    if(t(cont)%*%cont<err){
      index = 1
      break
    }
  }
  theta = theta0
  phi = optim(rep(0,m+1), ln_cox, Z1 = Z1, 
              Time = Time, theta = theta, D = D, 
              control = c(fnscale=-1))$par
  V1 = sort(Lambda0(Time,phi))
  var.theta = diag(solve(info/n))/n   
  sd.theta = sqrt(var.theta)            #the standard error
  z.score = theta*(var.theta)^(-1)*(theta)  #significance of one parameter
  p.va = sapply(z.score, pchisq, df = 1)
  p.value = 1-p.va   #significance
  #V2 = sort(Time)
  if(graph==T){
    plot(sort(Time),V1,type = "b")  #graph of Lambda0
    #lines(sort(Time),V2)
  }
  return(list(phi = phi, theta = theta, sd.theta = sd.theta, p.value = p.value, index = index))
}

######################################################################
#Newton-Raphson method for theta: expand grid of eta
estNMLE_nr1 = function(Z1, Z2, Time, D, m = 3, graph = F, err = 1e-5,M = 50){
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  theta0 = rep(0,d)  #initial value of theta
  eta0 = mean(Z2)   #initial value of eta
  #phi0 = rep(0,m+1)
  u = as.numeric(min(Time))
  v = as.numeric(max(Time))
  sez2 = sd(Z2)
  interval_eta = c(mean(Z2)-2*sez2, mean(Z2)+2*sez2)
  grid_eta = seq(interval_eta[1],interval_eta[2],by=0.1)
  index = 0
  #iterating profile estimation
  Z = cbind(Z1,Z1*(Z2>eta0),rep(1,n)*(Z2>eta0))
  #Berstein Polynominal
  B = c()
  for (s in 0:m){ 
    B = cbind(B,sapply(Time, bpoly,u = max(Time), 
                       v = min(Time),m = m, k = s))
  } 
  for(r in 1:M){
    #step 1: estimate Lambda0: Nelder-Mead Method with Berstein Polynominal
    phi = optim(rep(0,m+1), lnm, Z1 = Z1 , Z2 = Z2, D = D,
                Time = Time, theta = theta0, eta = eta0,
                control = c(fnscale=-1))$par   #parameter of baseline
    Lam = Lambda0(Time, phi)    #value of baseline hazard 
    
    #step 2: estimate theta: Newton-Raphson Method
    theta1 = theta0
    r0 = Lam*exp(Z%*%theta1)  #culmulative hazard function
    r1 = exp(-r0) #survival function
    score1 = -D*r0+(1-D)*(r1*r0)/(1-r1)
    score = t(t(score1)%*%Z)   #score function
    cong = score1^2
    info =  t(Z)%*%diag(as.vector(cong))%*%Z   #information
    invinfo = solve(info)
    theta0 = theta1 + invinfo%*%score
    
    
    #step 3: estimate eta, grid search
    eta1 = eta0
    efind = which.max(sapply(grid_eta, lnm, theta = theta0, Time = Time, Z1 = Z1,
                             Z2 = Z2, phi = phi, D = D))
    eta0 = grid_eta[efind]   #grid search 
    
    cont = c(theta0,eta0)-c(theta1,eta1)
    if(t(cont)%*%cont<err){   #whether convergece
      index = 1   #convergence
      break
    }
  }
  theta = as.vector(theta0)
  eta = eta0
  phi = optim(rep(0,m+1), lnm, Z1 = Z1 , Z2 = Z2, D = D,
              Time = Time, theta = theta, eta = eta,
              control = c(fnscale=-1))$par    #renew the baseline
  V1 = sort(Lambda0(Time,phi))
  var.theta = diag(solve(info/n))/n   
  sd.theta = sqrt(var.theta)            #the standard error
  z.score = theta*(var.theta)^(-1)*(theta)  #significance of one parameter
  p.va = sapply(z.score, pchisq, df = 1)
  p.value = 1-p.va   #significance
  #V2 = sort(Time)
  if(graph==T){
    plot(sort(Time),V1,type = "b")  #graph of baseline Lambda0
    #lines(sort(Time),V2)
  }
  return(list(phi = phi, theta = theta, sd.theta = sd.theta, p.value = p.value,
              eta = eta, index = index))
}

#######################################################################
#with linear control variable
#theta = c(alpha,beta1,beta0,gamma)
lnm_con = function(phi, Time, Z1, Z2, con, theta, eta, D){ 
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  q = ncol(con[1,])   #length(theta)=d+q
  m = length(phi)-1
  Z = cbind(Z1,Z1*(Z2>eta),rep(1,n)*(Z2>eta),con) #alpha,beta1,beta0,gamma
  h1 = as.matrix(exp(Z%*%theta)) #Z*theta
  B = c()
  for(s in 0:m){  #Berstein polynomials matrix
    B = cbind(B,sapply(Time, bpoly,u=max(Time),v=min(Time),m=m,k=s))
  }
  phi_s <- cumsum(exp(phi))
  A1 = as.matrix(B%*%phi_s)    #Lambda0n
  B1 = t(A1)%*%(diag(-D))%*%h1 #Censor
  G = function(u){
    return(log(1-exp(-u)))
  }
  #sapply(A1*h1 ,G)
  A2 = sapply(A1*h1, G)
  #A2 = replace(A2, A2 <1e-4 , 1e-4)
  #summary(A2)
  B2 = (1-D)%*%A2  #Not Censor
  return(B1+B2)
}

estNMLE_nrcon = function(Z1, Z2, Time, D, con, m = 3, graph = F, err = 1e-5,M = 50){
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  q = ncol(con)
  theta0 = rep(0,d+q)  #initial value of theta=c(alpha,beta1,beta0,gamma)
  eta0 = mean(Z2)   #initial value of eta
  #phi0 = rep(0,m+1)
  u = as.numeric(min(Time))
  v = as.numeric(max(Time))
  sez2 = sd(Z2)
  interval_eta = c(mean(Z2)-2*sez2, mean(Z2)+2*sez2)
  grid_eta = seq(interval_eta[1],interval_eta[2],by=0.1)
  index = 0
  #iterating profile estimation
  Z = cbind(Z1,Z1*(Z2>eta0),rep(1,n)*(Z2>eta0),con)
  #Berstein Polynominal
  B = c()
  for (s in 0:m){ 
    B = cbind(B,sapply(Time, bpoly,u = max(Time), 
                       v = min(Time),m = m, k = s))
  } 
  for(r in 1:M){
    #step 1: estimate Lambda0: Nelder-Mead Method with Berstein Polynominal
    phi = optim(rep(0,m+1), lnm_con, Z1 = Z1 , Z2 = Z2, D = D, con = con,
                Time = Time, theta = theta0, eta = eta0,
                control = c(fnscale=-1))$par   #parameter of baseline
    Lam = Lambda0(Time, phi)    #value of baseline hazard 
    
    #step 2: estimate theta: Newton-Raphson Method
    theta1 = theta0
    r0 = Lam*exp(Z%*%theta1)  #culmulative hazard function
    r1 = exp(-r0) #survival function
    score1 = -D*r0+(1-D)*(r1*r0)/(1-r1)
    score = t(t(score1)%*%Z)   #score function
    cong = score1^2
    info =  t(Z)%*%diag(as.vector(cong))%*%Z   #information
    invinfo = solve(info)
    theta0 = theta1 + invinfo%*%score
    
    
    #step 3: estimate eta, grid search
    eta1 = eta0
    efind = which.max(sapply(grid_eta, lnm_con, theta = theta0, Time = Time, Z1 = Z1,
                             Z2 = Z2, phi = phi, D = D, con = con))
    eta0 = grid_eta[efind]   #grid search 
    
    cont = c(theta0,eta0)-c(theta1,eta1)
    if(t(cont)%*%cont<err){   #whether convergece
      index = 1   #convergence
      break
    }
  }
  theta = as.vector(theta0)
  eta = eta0
  phi = optim(rep(0,m+1), lnm_con, Z1 = Z1 , Z2 = Z2, D = D, con = con,
              Time = Time, theta = theta, eta = eta,
              control = c(fnscale=-1))$par    #renew the baseline
  V1 = sort(Lambda0(Time,phi))
  var.theta = diag(solve(info/n))/n   
  sd.theta = sqrt(var.theta)            #the standard error
  z.score = theta*(var.theta)^(-1)*(theta)  #significance of one parameter
  p.va = sapply(z.score, pchisq, df = 1)
  p.value = 1-p.va   #significance
  #V2 = sort(Time)
  if(graph==T){
    plot(sort(Time),V1,type = "b")  #graph of baseline Lambda0
    #lines(sort(Time),V2)
  }
  return(list(phi = phi, theta = theta, sd.theta = sd.theta, 
              p.value = p.value, eta = eta, index = index))
}
###################################################################################################
#For a single Cox model without change point and control variable
ln_coxcon = function(phi, Time, D, Z1, theta, con){ 
  n = length(Z2)
  p = ncol(Z1)
  q = ncol(con)
  m = length(phi)-1
  #Z = cbind(Z1[,1:p],Z1[,1:p]*(Z2>eta),rep(1,n)*(Z2>eta))
  Z = cbind(Z1,con)
  h1 = as.matrix(exp(as.matrix(Z%*%theta))) #Z*theta
  B = c()
  for(s in 0:m){  #Berstein polynomials matrix
    B = cbind(B,sapply(Time, bpoly,u=max(Time),v=min(Time),m=m,k=s))
  }
  phi_s <- cumsum(exp(phi))
  A1 = as.matrix(B%*%phi_s)    #Lambda0n
  B1 = t(A1)%*%(diag(-D))%*%h1 #Censor
  G = function(u){
    return(log(1-exp(-u)))
  }
  #sapply(A1*h1 ,G)
  B2 = (1-D)%*%sapply(A1*h1, G)  #Not Censor
  return(B1+B2)
}


estNMLE_coxcon = function(Z1, Time, D, con, m = 3, graph = F, err = 1e-5,M = 50){
  n = nrow(Z1)
  p = ncol(Z1)
  q = ncol(con)
  theta0 = rep(0, p+q)  #initial value of theta
  #initial value of eta
  u = as.numeric(min(Time))
  v = as.numeric(max(Time))
  index = 0
  #Berstein Polynominal
  B = c()
  for (s in 0:m){ 
    B = cbind(B,sapply(Time, bpoly,u = max(Time), 
                       v = min(Time),m = m, k = s))
  } 
  #iterating profile estimation
  Z = cbind(Z1,con)
  for(r in 1:M){
    #step 1: estimate Lambda0: Nelder-Mead Method
    #Z = cbind(Z1[,1:p],Z1[,(p+1):(2*p)]*(Z2>eta0),rep(1,n)*(Z2>eta0))
    
    #log-likelihood w.r.t Lambda
    #grad
    #gradient(f=l_Lambda,x=c(phi),Time=Time, Z1=Z1, Z2=Z2, theta=theta, eta=eta)
    phi = optim(rep(0,m+1), ln_coxcon, Z1 = Z1, D = D, con = con,
                Time = Time, theta = theta0, 
                control = c(fnscale=-1))$par
    #Lambda0(Time,phi)  #value of baseline hazard
    #plot(sort(Time),sort(Lambda0(Time,phi)),type = "b")  #graph of Lambda0
    #lines(sort(Time),sort(Time))
    Lam = Lambda0(Time, phi)
    
    #step 2: estimate theta: Newton-Raphson Method
    theta1 = theta0
    r0 = Lam*exp(Z%*%theta1)  #culmulative hazard function
    r1 = exp(-r0) #survival function
    score1 = -D*r0+(1-D)*(r1*r0)/(1-r1)
    score = t(t(score1)%*%Z)   #score function
    cong = score1^2
    info =  t(Z)%*%diag(as.vector(cong))%*%Z   #information
    theta0 = theta1 + ginv(info)%*%score
    
    cont = theta0 - theta1
    if(t(cont)%*%cont<err){
      index = 1
      break
    }
  }
  theta = as.vector(theta0)
  phi = optim(rep(0,m+1), ln_coxcon, Z1 = Z1, con = con,
              Time = Time, theta = theta, D = D, 
              control = c(fnscale=-1))$par
  V1 = sort(Lambda0(Time,phi))
  var.theta = diag(solve(info/n))/n   #variance of the regression parameter
  se.theta = sqrt(var.theta)  #the standard error
  z.score = (theta)*(var.theta)^(-1)*theta
  p.va = sapply(z.score, FUN = pchisq, df = 1)
  p.value = 1-p.va
  #V2 = sort(Time)
  if(graph==T){
    plot(sort(Time),V1,type = "b")  #graph of Lambda0
    #lines(sort(Time),V2)
  }
  return(list(phi = phi, theta = theta, se.theta = se.theta, p.value = p.value, index = index))
}

