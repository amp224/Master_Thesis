coefs_from_2d_kernel = function(k, dims, w, b, g){
  # k: kernel size (1 or 2)
  # dims: size of the 2D signal in a vector (n,m)
  # w: matrix of weights (kernel) of size kxk
  # b: bias of the kernel
  # g: vector of length q+1 containing g(0) and up to g^{(q)}(0)/q!
  
  q = length(g) - 1
  
  # poly will be a list of lists of lists
  # the i-th element of poly is a list representing the i-th row of the matrix
  # the j-th element of the i-th list will contain the
  # coefficients and labels for the (i,j) element of the feature map
  poly = vector(mode='list', length=dims[1]-k+1)
  row = vector(mode='list', length=dims[2]-k+1)
  
  # for kernel size k=1 and any Taylor order
  if (k==1){
    
    # each row of the feature map
    for (i in 1:length(poly)){
      # each element in the i-th row
      for (j in 1:length(row)){
        coefs = c(0)
        labels = c("0")
        
        # first the intercept to Taylor order q
        for (l in 0:q){
          coefs[1] = coefs[1] + g[l+1]*b^l
        }
        
        # counter variable
        count = 1
        
        coefs_D = rep(0,q)
        labels_D = rep(paste(i,j,sep=","), q)
        
        # D = 1 (first order coefficient)
        for (l in 1:q){
          coefs_D[1] = coefs_D[1] + g[l+1]*factorial(l)/(factorial(1)*factorial(l-1))*b^(l-1)*w
        }
        
        # higher order coefficients
        for (D in 2:q){
          # general formula
          for (l in D:q){
            coefs_D[D] = coefs_D[D] + g[l+1]*factorial(l)/(factorial(D)*factorial(l-D))*b^(l-D)*w^D
          }
          labels_D[D] = paste(paste(i,j,sep=","), labels_D[D-1], sep=":")
        }
        coefs = c(coefs, coefs_D)
        labels = c(labels, labels_D)
        
        row[[j]] = list(coefs=coefs, labels=labels)
      }
      poly[[i]] = row
    }
    return (poly)
  }
  
  # for Taylor order q=2 and kernel size k=2
  else if (k==2 & q==2){
    
    # each row of the feature map
    for (i in 1:length(poly)){
      # each element in the i-th row
      for (j in 1:length(row)){
        coefs = c(0)
        labels = c("0")
        
        # intercept to Taylor order 2
        for (l in 0:2){
          coefs[1] = coefs[1] + g[l+1]*b^l
        }
        
        # counter
        count = 1
        
        # first degree coefficients
        for (r in 1:2){
          for (s in 1:2){
            count = count + 1
            coefs[count] = w[r,s] + 2*b*w[r,s]
            labels[count] = paste(as.character(c(r+i-1, s+j-1)), 
                                  collapse=",")
          }
        }
        
        # second degree (t_1 t_2)(t_1 t_2) coefficients
        for (r in 1:2){
          for (s in 1:2){
            count = count + 1
            coefs[count] = w[r,s]^2
            labels[count] = paste0(paste(as.character(c(r+i-1,s+j-1)), 
                                         collapse=","),
                                   ":",
                                   paste(as.character(c(r+i-1,s+j-1)), 
                                         collapse=","))
          }
        }
        
        aux = 0
        
        # second degree (t_1 t_2) (t_3 t_4) coefficients
        for (r in 1:2){
          for (s in 1:2){
            count = count + 1
            for (n in 1:2){
              for (m in 1:2){
                if (r<n || s<m){
                  coefs[count] = 2*w[r,s]*w[n,m]
                  labels[count] = paste0( 
                    paste(as.character(c(r+i-1,s+j-1)), 
                          collapse=","),
                    ":",
                    
                    paste(as.character(c(n+i-1,m+j-1)), 
                          collapse=",")
                    
                  )
                }
              }
            }
          }
        }
        
        row[[j]] = list(coefs=coefs, labels=labels)
      }
      poly[[i]] = row
    }
    return (poly)
  }
  
  # if k>3 or k=2 and q>2, there is no formula for the coefficients (yet)
  else{
    message("Error: This combination of kernel size and Taylor order has not been implemented.")
  }

}
