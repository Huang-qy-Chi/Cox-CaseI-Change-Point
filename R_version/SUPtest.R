source("bpoly.R")
source("estNMLE.R")
library(MASS)
SUP = function(Z1, Z2, D, Time, k=5, alpha=0.05, B=100){
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  Z = as.matrix(Z1)
  Z2max = max(Z2)
  Z2min = min(Z2)
  step = (Z2max-Z2min)/k
  eta_grid = seq(Z2min,Z2max,step)
  res = estNMLE_cox(Z, Time, D)
  alpha0 = res$theta      #Cox estimation of regression parameter without change point
  theta = c(alpha0,rep(0,p+1))   #Under H0, beta=0
  phi0 = res$phi
  Lambda = Lambda0(Time, phi0)    #baseline estimation without change point 
  
  #eta = 2
  #lambda = Lambda
  #test statistics
  #sup = 0
  ## calculate one single statistics for a given set of (Z1,Z2,D,theta,lambda,eta)
  supstat = function(Z1,Z2,D,theta,lambda,eta){
    n = length(Z2)
    p = ncol(Z1)
    d = 2*p+1
    Z = as.matrix(Z1[,1:p])
    ZZ = cbind(Z1[,1:p],rep(1,n))
    theta1 = theta[1:p]
    r0 = lambda*exp(Z%*%theta1)  #cumulative hazard function under beta=0
    r1 = exp(-r0) #survival function
    sup = c()
    for(eta in eta){
      score1 = (-D*r0+(1-D)*(r1*r0)/(1-r1))*(Z2>eta)
      score = t(t(score1)%*%ZZ)   #score function
      cong = score1^2
      info =  t(ZZ)%*%diag(as.vector(cong))%*%ZZ 
      sup = c(sup,t(score)%*%ginv(info)%*%score)
    }
    return(max(sup))
  }
  sup = supstat(eta = eta_grid, Z1 = Z1, Z2 = Z2, D = D,
                      theta = theta, lambda = Lambda)
  
  #distribution and quantile
  sup_quant = function(s){
    Z2_boot = sample(Z2, n)  #permutation
    sup_rec = supstat(eta = eta_grid, Z1 = Z1, Z2 = Z2_boot, D = D,
                      theta = theta, lambda = Lambda)
    return(sup_rec)
  }
  quantile = sapply(1:B, sup_quant)
  alpha_quan = sort(quantile)[B*(1-alpha)]
  index = sup>alpha_quan  #if reject, index = 1 or True
  pquan = abs(sort(quantile)-sup)
  p.va = 1-which.min(pquan)/B  #quasi p-value
  return(list(index = index, sup_stat = sup,
              quantile = alpha_quan, p.value = p.va) )
}

SUPpar = function(Z1, Z2, D, Time, k=5, alpha=0.05, B=1000){
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  Z = as.matrix(Z1)
  Z2max = max(Z2)
  Z2min = min(Z2)
  step = (Z2max-Z2min)/k
  eta_grid = seq(Z2min,Z2max,step)
  res = estNMLE_cox(Z, Time, D)
  alpha0 = res$theta      #Cox estimation of regression parameter without change point
  theta = c(alpha0,rep(0,p+1))   #Under H0, beta=0
  phi0 = res$phi
  Lambda = Lambda0(Time, phi0)    #baseline estimation without change point 
  
  #eta = 2
  #lambda = Lambda
  #test statistics
  #sup = 0
  ## calculate one single statistics for a given set of (Z1,Z2,D,theta,lambda,eta)
  supstat = function(Z1,Z2,D,theta,lambda,eta){
    n = length(Z2)
    p = ncol(Z1)
    d = 2*p+1
    Z = as.matrix(Z1[,1:p])
    ZZ = cbind(Z1[,1:p],rep(1,n))
    theta1 = theta[1:p]
    r0 = lambda*exp(Z%*%theta1)  #cumulative hazard function under beta=0
    r1 = exp(-r0) #survival function
    sup = c()
    for(eta in eta){
      score1 = (-D*r0+(1-D)*(r1*r0)/(1-r1))*(Z2>eta)
      score = t(t(score1)%*%ZZ)   #score function
      cong = score1^2
      info =  t(ZZ)%*%diag(as.vector(cong))%*%ZZ 
      sup = c(sup,t(score)%*%ginv(info)%*%score)
    }
    return(max(sup))
  }
  sup = supstat(eta = eta_grid, Z1 = Z1, Z2 = Z2, D = D,
                theta = theta, lambda = Lambda)
  
  #distribution and quantile
  sup_quant = function(s){
    Z2_boot = sample(Z2, n)  #permutation
    sup_rec = supstat(eta = eta_grid, Z1 = Z1, Z2 = Z2_boot, D = D,
                      theta = theta, lambda = Lambda)
    return(sup_rec)
  }
  library(doParallel)
  library(foreach)
  v1 = detectCores()
  cl <- makeCluster(v1-2)
  registerDoParallel(cl)
  quantile <- foreach(i=1:B,.combine = c,.packages = c("LaplacesDemon","optimx","rootSolve","expm","pracma","MASS"))%dopar%{
    sup_quant(i)
  }
  stopCluster(cl)
  quantile = unlist(quantile)
  alpha_quan = sort(quantile)[B*(1-alpha)]
  index = sup>alpha_quan  #if reject, index = 1 or True
  pquan = abs(sort(quantile)-sup)
  p.va = 1-which.min(pquan)/B  #quasi p-value
  return(list(index = index, sup_stat = sup,
              quantile = alpha_quan, p.value = p.va) )
}



SUP_con = function(Z1, Z2, D, Time, con, k=5, alpha=0.05, B=100){
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  q = ncol(con)
  Z = as.matrix(Z1)
  Z2max = max(Z2)
  Z2min = min(Z2)
  step = (Z2max-Z2min)/k
  eta_grid = seq(Z2min,Z2max,step)
  res = estNMLE_coxcon(Z, Time, D, con)
  alpha0 = res$theta[1:p]  #Cox estimation of regression parameter without change point
  gamma0 = res$theta[(p+1):(p+q)]
  theta = c(alpha0,rep(0,p+1),gamma0)   #Under H0, beta=0
  phi0 = res$phi
  Lambda = Lambda0(Time, phi0)    #baseline estimation without change point 
  
  #eta = 2
  #lambda = Lambda
  #test statistics
  #sup = 0
  ## calculate one single statistics for a given set of (Z1,Z2,D,theta,lambda,eta)
  supstat = function(Z1,Z2,con,D,theta,lambda,eta){
    n = length(Z2)
    p = ncol(Z1)
    d = 2*p+1
    q = ncol(con)
    Z = cbind(as.matrix(Z1),con)
    ZZ = cbind(Z1,rep(1,n))
    alpha0 = theta[1:p]  
    gamma0 = theta[(p+1):(p+q)]
    theta1 = c(alpha0,gamma0)
    r0 = lambda*exp(Z%*%theta1)  #cumulative hazard function under beta=0
    r1 = exp(-r0) #survival function
    sup = c()
    for(eta in eta){
      score1 = (-D*r0+(1-D)*(r1*r0)/(1-r1))*(Z2>eta)
      score = t(t(score1)%*%ZZ)   #score function
      cong = score1^2
      info =  t(ZZ)%*%diag(as.vector(cong))%*%ZZ 
      sup = c(sup,t(score)%*%ginv(info)%*%score)
    }
    return(max(sup))
  }
  sup = supstat(eta = eta_grid, Z1 = Z1, Z2 = Z2, D = D,
                theta = theta,con = con, lambda = Lambda)
  
  #distribution and quantile
  sup_quant = function(s){
    Z2_boot = sample(Z2, n)  #permutation
    sup_rec = supstat(eta = eta_grid, Z1 = Z1, Z2 = Z2_boot, D = D,
                      con = con,theta = theta, lambda = Lambda)
    return(sup_rec)
  }
  quantile = sapply(1:B, sup_quant)
  alpha_quan = sort(quantile)[B*(1-alpha)]
  index = sup>alpha_quan  #if reject, index = 1/True
  pquan = abs(sort(quantile)-sup)
  p.va = 1-which.min(pquan)/B  #quasi p-value
  return(list(index = index, sup_stat = sup, 
              quantile = alpha_quan, p.value = p.va) )
}



SUP_conpar = function(Z1, Z2, D, Time, con, k=5, alpha=0.05, B=1000){
  n = length(Z2)
  p = ncol(Z1)
  d = 2*p+1
  q = ncol(con)
  Z = as.matrix(Z1)
  Z2max = max(Z2)
  Z2min = min(Z2)
  step = (Z2max-Z2min)/k
  eta_grid = seq(Z2min,Z2max,step)
  res = estNMLE_coxcon(Z, Time, D, con)
  alpha0 = res$theta[1:p]  #Cox estimation of regression parameter without change point
  gamma0 = res$theta[(p+1):(p+q)]
  theta = c(alpha0,rep(0,p+1),gamma0)   #Under H0, beta=0
  phi0 = res$phi
  Lambda = Lambda0(Time, phi0)    #baseline estimation without change point 
  
  #eta = 2
  #lambda = Lambda
  #test statistics
  #sup = 0
  ## calculate one single statistics for a given set of (Z1,Z2,D,theta,lambda,eta)
  library(MASS)
  supstat = function(Z1,Z2,con,D,theta,lambda,eta){
    n = length(Z2)
    p = ncol(Z1)
    d = 2*p+1
    q = ncol(con)
    Z = cbind(as.matrix(Z1),con)
    ZZ = cbind(Z1,rep(1,n))
    alpha0 = theta[1:p]  
    gamma0 = theta[(p+1):(p+q)]
    theta1 = c(alpha0,gamma0)
    r0 = lambda*exp(Z%*%theta1)  #cumulative hazard function under beta=0
    r1 = exp(-r0) #survival function
    sup = c()
    for(eta in eta){
      score1 = (-D*r0+(1-D)*(r1*r0)/(1-r1))*(Z2>eta)
      score = t(t(score1)%*%ZZ)   #score function
      cong = score1^2
      info =  t(ZZ)%*%diag(as.vector(cong))%*%ZZ 
      sup = c(sup,t(score)%*%ginv(info)%*%score)
    }
    return(max(sup))
  }
  sup = supstat(eta = eta_grid, Z1 = Z1, Z2 = Z2, D = D,
                theta = theta,con = con, lambda = Lambda)
  
  #distribution and quantile
  sup_quant = function(s){
    Z2_boot = sample(Z2, n)  #permutation
    sup_rec = supstat(eta = eta_grid, Z1 = Z1, Z2 = Z2_boot, D = D,
                      con = con,theta = theta, lambda = Lambda)
    return(sup_rec)
  }
  library(doParallel)
  library(foreach)
  v1 = detectCores()
  cl <- makeCluster(v1-2)
  registerDoParallel(cl)
  quantile <- foreach(i=1:B,.combine = c,.packages = c("LaplacesDemon","optimx","rootSolve","expm","pracma","MASS"))%dopar%{
    sup_quant(i)
  }
  stopCluster(cl)
  quantile = unlist(quantile)
  alpha_quan = sort(quantile)[B*(1-alpha)]
  index = sup>alpha_quan  #if reject, index = 1/True
  pquan = abs(sort(quantile)-sup)
  p.va = 1-which.min(pquan)/B  #quasi p-value
  return(list(index = index, sup_stat = sup, 
              quantile = alpha_quan, p.value = p.va) )
}



























































