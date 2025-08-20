library(parallel)

#m out of n bootstrap
##WARNING: Very SLOW! Avoid to use!
source("bpoly.R")
source("estNMLE.R")
mn_boot = function(Z1,Z2,D,Time,B=100, q=20,alpha=0.05){
  n = length(Z2)
  d = ncol(Z1)
  p = (d-1)/2
  M = floor(n*((q-3):(q-1)/q)) #sample size
  eta.n = estNMLE_nr(Z1,Z2,Time,D)$eta #full-data estimation of eta
  x = seq(min(Z2),max(Z2),by = 0.5)
  
  bot1 = function(m,Z1,Z2,D,Time,eta.n,x){
    u = length(x)
    rw = rep(0,B)
    for(s in 1:B){
      loc = sample(1:n,m)  #m out of n bootstarp
      Z1.b = Z1[loc,]
      Z2.b = Z2[loc]
      D.b = D[loc]
      Time.b = Time[loc]
      rw[s] = estNMLE_nr(Z1.b,Z2.b,Time.b,D.b)$eta
    }
    diff_eta = rw-eta.n
    abseta = sort(diff_eta)[m*(1-alpha)]
    rec = matrix(rep(0,m*u),ncol = u)
    for(j in 1:u){
      rec[,j] = m*diff_eta<x
    }
    ecdf = rowMeans(rec) #value of ecdf
    return(list(ecdf = ecdf, Q.alpha = abseta))
  }
  u = length(x)
  v = length(M)
  CDF = matrix(rep(0,u*v),ncol = u)
  dCDF = matrix(rep(0,u*(v-1)),ncol = u)
  for(m in M){
    CDF[m,] = bot1(m,Z1,Z2,D,Time,eta.n,x)$ecdf
  }
  for(l in 1:(m-1)){
    dCDF[l,] = CDF[l+1,]-CDF[l,]
  }
  dCDF = abs(dCDF)
  uo = rep(0,v)
  for(t in 1:(m-1)){
    uo[t] = max(dCDF[t,]) #sup_x|cdf_m-cdf_n|
  }
  q_sort = which.min(uo)
  m = n*(q_sort/q)
  Q.alp = bot1(m,Z1,Z2,D,Time,eta.n,x)$Q.alpha
  interval = c(eta.n-(Q.alp*n)/m, eta.n+(Q.alp*n)/m)
  return(list(m = m, Q.alp = Q.alp, interval = interval))
}






















