evaluate_poly_1d = function(x, poly){
  # x: input signal
  # poly: List of lists, each containing the coefs and the labels 
  #       of the polynomial for each element of the feature map
  
  # output should have as many elements as the feature map
  out = c()
  
  # for each element of the feature map
  for (i in 1:length(poly)){
    # temporarily save the element's coefficients with labels
    coefs = poly[[i]]$coefs
    labels = poly[[i]]$labels
    
    # first the intercept
    out[i] = coefs[1]
    
    # for each possible coefficient
    for (j in 2:length(coefs)){
      variable_indices = as.integer(unlist(strsplit(labels[j], ",")))
      
      product = 1
      
      # for each combination of variables
      for (k in 1:length(variable_indices)){
        # calculate the corresponding product
        product = product*x[variable_indices[k]]
      }
      # add until the whole polynomial has been evaluated
      out[i] = out[i] + coefs[j]*product
    }
  }
  return(out)
}
