rm(list = ls())
setwd("C:\\Users\\hqysd\\Desktop\\CaseICP\\Rcode\\Cox-CaseI-Change-Point-main")
source("bpoly.R")
source("gendata.R")
source("estNMLE.R")
source("SUPtest.R")
library(doParallel)
library(foreach)
data1 = gendata(500,0.5,stepsize = 0.1)
Z1 = data1$Z1    #parameter: alpha, beta_0, beta_1 
Z2 = data1$Z2    #change point
Time = data1$Time  #survival time
D = data1$D    #censoring indicator
estNMLE(Z1,Z2,Time,D,m=3,graph=T,err=1e-5)
summary(Time)

data1 = gendata_m(n = 500,pr = 0.5,rho = 0.5,
                  theta = c(-1,0,1,-1,2),stepsize = 0.1, eta = 2)
Z1 = data1$Z1    #parameter: alpha, beta_0, beta_1 
Z2 = data1$Z2    #change point
Time = data1$Time  #survival time
D = data1$D    #censoring indicator
estNMLE_m(Z1, Z2, Time, D, m = 3, graph = T, err = 1e-5, M = 50) #using optim()
estNMLE_nr(Z1, Z2, Time, D, m = 3, graph = T, err = 1e-5, M = 50)  #using Newton-Raphson
estNMLE_cox(Z1[,1:2], Time, D, m = 3, graph = T, err = 1e-5, M = 50)  #using Newton-Raphson



data1 = gendata_m(n = 500,pr = 0.5,rho = 0.5,
                  theta = c(-1,0,0),stepsize = 0.1, eta = 2)
Z1 = data1$Z1    #parameter: alpha, beta_0, beta_1 
Z2 = data1$Z2    #change point
Time = data1$Time  #survival time
D = data1$D    #censoring indicator
#estNMLE_nr(Z1, Z2, Time, D, m = 3, graph = T, err = 1e-5, M = 50) 
#estNMLE_cox(as.matrix(Z1[,1:1]), Time, D, m = 3, graph = T, err = 1e-5, M = 50)
SUP(Z1,Z2,D,Time,k=20,alpha = 0.05)
#summary(Time)
#head(Z1)


RR = rep(0,500)

v1 = detectCores()
cl <- makeCluster(v1-2)
registerDoParallel(cl)
RR <- foreach(i=1:500,.combine = c,.packages = "optimx")%dopar%{
  data1 = gendata(500,0.5,stepsize = 0.1)
  Z1 = data1$Z1
  Z2 = data1$Z2
  Time = data1$Time
  D = data1$D
  estNMLE(Z1,Z2,Time,D,m=3,graph=F,err=1e-5)$theta[2]
}
mean(RR)-1.5
stopImplicitCluster()
stopCluster(cl)


v1 = detectCores()
cl <- makeCluster(v1-6)
registerDoParallel(cl)
RR <- foreach(i=1:500,.combine = c,.packages = c("LaplacesDemon","optimx","rootSolve","expm","pracma","MASS"))%dopar%{
  data1 = gendata_m(n = 500,pr = 0.5,rho = 0.5,
                    theta = c(-1,0,0),stepsize = 0.1, eta = 2)
  Z1 = data1$Z1    #parameter: alpha, beta_0, beta_1 
  Z2 = data1$Z2    #change point
  Time = data1$Time  #survival time
  D = data1$D    #censoring indicator
  SUP(Z1,Z2,D,Time,k=10,alpha = 0.05)$index
}
mean(RR)
stopImplicitCluster()
stopCluster(cl)




