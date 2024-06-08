bpoly=function(t,k,m,u,v){
  a = choose(m,k)      #组合数
  b = ((t-v)/(u-v))^k    
  c = (1-(t-v)/(u-v))^(m-k)
  return(a*b*c)
}

