coefs_from_1d_kernel = function(n, w, b, g){
  # n: size (length) of the 1D signal
  # w: vector of weights (kernel) of length k
  # b: bias of the kernel 
  # g: vector of length q+1 containing g(0) and up to g^{(q)}(0)/q!
  
  q = length(g) - 1
  k = length(w)
  
  # j-th element of poly contains the coefficients and labels of the j-th element of the feature map
  # the feature map has length n-k+1 after 'valid' convolution
  poly = vector(mode="list", length=n-k+1)

  
  for (j in 1:(n-k+1)){
    # coefficients and labels for each element of the feature map
    coefs = c(0)
    labels = c("0")
    
    # intercept 
    for (l in 0:q){
      coefs[1] = coefs[1] + g[l+1]*b^l
    }
  
    # higher order coefficients
    for (D in 1:q){
      # the combinations for the j-th element can only contain variables from x_j to x_{j+k-1}
      indices = combinations(k, D, v=j:(j+k-1), repeats.allowed=T)
      indices.rows = nrow(indices) # total number of combinations
      
      # to temporarily storage coefficients and labels for order D
      coefs_D = rep(0,indices.rows)
      labels_D = rep("label",indices.rows)
    
      # over all possible combinations
      for (combination_index in 1:indices.rows){
        
        # vector of the combination with indices t_1 t_2 ... t_D
        t_values = indices[combination_index,]
        
        # label of the combination in string form "t_1,t_2,...,t_D"
        labels_D[combination_index] = paste(as.character(indices[combination_index,]),
                                            collapse = ",")
      
        # vector (p_1,...,p_k) with number of times each variable appears in the combination
        p = rep(0,k)
        for (i in j:(j+k-1)){
          p[i-j+1] = sum(t_values == i)
        }
      
        aux = 0
        
        # general formula
        for (l in D:q){
          aux = aux + g[l+1]/prod(factorial(p))*b^(l-D)*prod(w^p)
        }
        coefs_D[combination_index] = aux
      }
      # save the D-th coefficients and labels
      coefs = c(coefs, coefs_D)
      labels = c(labels,labels_D)
    }
    # save the j-th element's polynomial
    poly[[j]] = list(coefs=coefs, labels=labels)
  }
  
  return(poly)
}
