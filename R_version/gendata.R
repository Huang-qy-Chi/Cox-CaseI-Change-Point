library(MASS)
library(expm)
library(LaplacesDemon)

#Find a proper c for generating the censoring time w.r.t. right-censoring rate
search_c = function(pr, B=1000, stepsize = 0.1, theta = c(-1,1.5,0.5), eta=2){
  Z2 = rnorm(B,eta,eta-0.5)  #change point
  z = rnorm(B,0,1)
  Z1 = cbind(z,z*(Z2>eta),rep(1,B)*(Z2>eta))   #data
  #is.matrix(Z1)
  u = runif(B,0,1)
  st = -log(u)/(exp(Z1%*%theta)) #generate real survival time
  c_ser = seq(0,max(st),by = stepsize)  #grid search for c
  emp_cen = function(m){
    C.vec <-  runif(B, 0, m)
    cenRate <- mean(C.vec < st)
    return(cenRate)
  }
  rec = which.min(abs(sapply(c_ser,emp_cen)-pr))  #simulated censoring rate
  return(c_ser[rec])
}

gendata = function(n, pr, theta = c(-1,1.5,0.5), eta=2, stepsize = 0.1){
  p = (length(theta)-1)/2
  Z2 = rnorm(n,eta,eta-0.5)  #change point
  z = rnorm(n,0,1)
  Z_record = cbind(z,z,rep(1,n))
  Z1 = cbind(z,z*(Z2>eta),rep(1,n)*(Z2>eta))   #data
  #is.matrix(Z1)
  c = mean(sapply(rep(pr,10),search_c,stepsize = stepsize)) #search for c
  u1 = runif(n,0,c)  #censoring time
  Trec = -log(runif(n,0,1))/(exp(Z1%*%theta))  #real survival time:Lambda0(t)=t
  D = as.numeric(Trec>u1)  #censoring indicator, D=1 means censoring
  #Time = pmin(Trec, u1)  # right censor observed time
  Time = u1  #case I interval: current status data
  return(list(Time = Time, D = D, Z1 = Z_record, Z2 = Z2))
}

library(LaplacesDemon)
gendata_m = function(n, pr, rho, theta = c(-1,1,1.5,-1.5,0.5), eta=2, stepsize = 0.1){
  p = (length(theta)-1)/2
  Z2 = rnorm(n,eta,eta-0.5)  #change point
  Sigma = rho*matrix(rep(1,p^2), ncol = p)+(1-rho)*diag(p)
  z = rmvn(n,rep(0,p),Sigma)
  Z_record = cbind(z,z,rep(1,n))
  Z1 = cbind(z,z*(Z2>eta),rep(1,n)*(Z2>eta))   #data
  #is.matrix(Z1)
  c = mean(sapply(rep(pr,10),search_c,stepsize = stepsize)) #search for c
  u1 = runif(n,0,c)  #censoring time
  Trec = -log(runif(n,0,1))/(exp(Z1%*%theta))  #real survival time:Lambda0(t)=t
  D = as.numeric(Trec>u1)  #censoring indicator, D=1 means censoring
  #Time = pmin(Trec, u1)  # right censor observed time
  Time = u1  #case I interval: current status data
  return(list(Time = Time, D = D, Z1 = z, Z2 = Z2))
}

gendata_con = function(n, pr, rho, theta = c(-1,1,1.5,-1.5,0.5), gamma = c(-1,1), eta=2, stepsize = 0.1){
  p = (length(theta)-1)/2
  Z2 = rnorm(n,eta,eta-0.5)  #change point
  Sigma = rho*matrix(rep(1,p^2), ncol = p)+(1-rho)*diag(p)
  Z1 = rmvn(n,rep(0,p),Sigma)
  q = length(gamma)
  Sigma1 = rho*matrix(rep(1,q^2), ncol = q)+(1-rho)*diag(q)
  con = rmvn(n,rep(0,q),Sigma1)
  #Z_record = cbind(z,z,rep(1,n))
  Z = cbind(Z1,Z1*(Z2>eta),rep(1,n)*(Z2>eta),con)   #data
  theta1 = c(theta,gamma)
  #is.matrix(Z1)
  c = mean(sapply(rep(pr,10),search_c,stepsize = stepsize)) #search for c
  u1 = runif(n,0,c)  #censoring time
  Trec = -log(runif(n,0,1))/(exp(Z%*%theta1))  #real survival time:Lambda0(t)=t
  D = as.numeric(Trec>u1)  #censoring indicator, D=1 means censoring
  #Time = pmin(Trec, u1)  # right censor observed time
  Time = u1  # current status data
  return(list(Time = Time, D = D, Z1 = Z1, Z2 = Z2, con = con))
}
