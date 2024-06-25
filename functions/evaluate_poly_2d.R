evaluate_poly_2d = function(k, x, poly){
  # k: kernel size (1 or 2)
  # x: input image
  # poly: List of lists of lists
  #       the j-th element of the i-th list containing the coefs and labels
  #       of the polynomial associated with the (i,j) element of the
  #       feature map
  
  # dimensions of the nxm feature map 
  n = length(poly)
  m = length(poly[[1]])
  # output has the dimensions of the feature map
  out = array(dim=c(n,m))
  
  # if 1x1 (general formula)
  if (k==1){
    
   # for each element of the feature map
    for (i in 1:n){
      for (j in 1:m){
        # temporarily save the element's coefficients and labels
        coefs = poly[[i]][[j]]$coefs
        labels = poly[[i]][[j]]$labels
        
        # first the intercept
        out[i,j] = coefs[1]
        
        # for each possible coefficient
        for (k in 2:length(coefs)){
          # the index within the coefficient vector
          # indicates the degree of the coefficient
          # (starting at 0)
          out[i,j] = out[i,j] + coefs[k]*x[i,j]^(k-1)
        }
      }
    }
    return (out)
  }
  
  # if 2x2 kernel (no general formula for any Taylor order, only up to q=2)
  else if (k==2){
    
    # for each element of the feature map
    for (i in 1:n){
      for (j in 1:m){
        # temporarily save the element's coefficients and labels
        coefs = poly[[i]][[j]]$coefs
        labels = poly[[i]][[j]]$labels
        
        # first the intercept
        out[i,j] = coefs[1]
        
        # for each possible coefficient
        for (k in 2:length(coefs)){
          indices = as.integer(unlist((strsplit(unlist(strsplit(labels[k], ":")), ","))))
          
          # if its a first degree coefficient (t1 t2)
          if (length(indices) == 2){
            out[i,j] = out[i,j] + coefs[k]*x[i,j]
          }
          
          # if it is a second degree coefficient 
          else{
            # if its a coefficient of the type (t1 t2)(t1 t2)
            if (indices[1]==indices[3] & indices[2]==indices[4]){
              out[i,j] = out[i,j] + coefs[k]*x[indices[1],indices[2]]^2
            }
            # if its of the type (t1 t2) (t3 t4)
            else{
              out[i,j] = out[i,j] + coefs[k]*x[indices[1],indices[2]]*x[indices[3],indices[4]]
            }
          }
        }
      }
    }
    return (out)
  }
  
  else{
    message("Error: The given kernel size has not been implemented.")
  }

}
