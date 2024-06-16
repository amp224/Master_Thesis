generate_data = function(case, n_sample, p, k, mean_range){
  # case: string indicating if 1D or 2D data
  # n_sample: number of signals to be generated
  # p: if 1D length of the signal, if 2D a square pxp signal is assumed
  # k: kernel size. If case="2D" a kxk kernel is assumed.
  # mean_range: vector of size 2 containing the range of the mean of the normal
  #             to be generated
  
  
  if (case=="1D"){
    data = array(dim=c(n_sample, p))
    target = c()
    mu = runif(p, mean_range[1], mean_range[2])
    for (i in 1:p){
      data[,i] = rnorm(n_sample, mean=mu[i], sd=1)
    }
    
    for (j in 1:n_sample){
      # linear combination coefficients
      lc = runif(p)
      # the target will be a linear combination of the variables
      target[j] = sum(lc*data[j,])
    }
  }
  
  else if (case=="2D"){
    data = array(dim=c(n_sample, p, p))
    target = c()
    mu = runif(p^2, mean_range[1], mean_range[2])
    mu = array(mu, dim=c(p,p))
    for (i in 1:p){
      for (j in 1:p){
        data[,i,j] = rnorm(n_sample, mean=mu[i,j], sd=1)
      }
    }
    for (k in 1:n_sample){
     lc = runif(p^2)
     target[k] = sum(lc*as.vector(data[k,,]))
    }
  }
  return(list(X=data, Y=target))
}
