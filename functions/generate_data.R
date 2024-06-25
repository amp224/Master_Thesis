generate_data = function(case, n_sample, p, k, mean_range){
  # case: string indicating if 1D or 2D data
  # n_sample: number of signals to be generated
  # p: if 1D length of the signal, if 2D a square pxp signal is assumed
  # k: kernel size. If case="2D" a kxk kernel is assumed.
  # mean_range: vector of size 2 containing the range for the means of the normals
  #             to be used
  
  # 1D signals (audio)
  if (case=="1D"){
    # array to store the signals
    data = array(dim=c(n_sample, p))
    # vector to store the target values
    target = c()
    # the elements in the same place in the signals come from the same distribution
    # generate as many means as elements in a signal
    mu = runif(p, mean_range[1], mean_range[2])
    for (i in 1:p){
      # generate the signals
      data[,i] = rnorm(n_sample, mean=mu[i], sd=1)
    }
    
    # generate linear combination coefficients
    lc = runif(p)
    
    for (j in 1:n_sample){
      # the target will be a linear combination of the variables
      target[j] = sum(lc*data[j,])
    }
  }
  
  # 2D signals (images)
  else if (case=="2D"){
    # array to store the images
    data = array(dim=c(n_sample, p, p))
    # vector to contain the target values
    target = c()
    # the elements in the same position in the matrix will come from the 
    # same distribution
    # generate as many means as pixels
    mu = runif(p^2, mean_range[1], mean_range[2])
    # reshape the means to match the images (useful later)
    mu = array(mu, dim=c(p,p))
    for (i in 1:p){
      for (j in 1:p){
        # generate data pixel by pixel
        data[,i,j] = rnorm(n_sample, mean=mu[i,j], sd=1)
      }
    }
    # generate the coefficients of the linear combination
    lc = runif(p^2)
    for (k in 1:n_sample){
      # compute the linear combination
      target[k] = sum(lc*as.vector(data[k,,]))
    }
  }
  # return the signals and target in a list
  return(list(X=data, Y=target))
}
