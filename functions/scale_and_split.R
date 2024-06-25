scale_and_split = function(case, data, split_prop=0.75){
  # MinMax scales the data to the [-1,1] range and
  # splits the data into train and test partitions
  
  # case: whether one-dimensional or two-dimensional data (string)
  # data: input data in list format with X the variables and Y the target
  # split_prop: proportion of data going to the train partition (float)
  
  
  X = data$X
  Y = data$Y
  
  if (case=="1D"){
    # save the maxima and minima of each variable (column-wise)
    maxs_X = apply(X, 2, max)
    mins_X = apply(X, 2, min)
  
    # scales the variables to [-1,1]
    X.scaled = scale(X, center=mins_X+0.5*(maxs_X-mins_X), 
                     scale=0.5*(maxs_X-mins_X))
    
    # carries out the partition
    part = sample(1:nrow(X), round(split_prop*nrow(X)))
  
    train_X = X.scaled[part,]
    test_X = X.scaled[-part,]
  }
  
  if (case=="2D"){
    n_sample = dim(X)[1]
    X.scaled = array(dim = dim(X))
    
    maxs_X = apply(X, c(2,3), max)
    mins_X = apply(X, c(2,3), min)

    for (i in 1:n_sample){
      # scales the variables to [-1,1]
      X.scaled[i,,] = (X[i,,] - mins_X-0.5*(maxs_X-mins_X))/(0.5*(maxs_X-mins_X))
    }
    # carries out the partition
    part = sample(1:n_sample, round(split_prop*n_sample))
    
    train_X = X.scaled[part,,]
    test_X = X.scaled[-part,,]
  }
  
  # save the maximum and minimum of the target
  max_Y = max(Y)
  min_Y = min(Y)
  
  # scales the target to [-1,1]
  Y.scaled = scale(Y, center=min_Y+0.5*(max_Y-min_Y), scale=0.5*(max_Y-min_Y))
  
  # perform the paritition on the target
  train_Y = Y.scaled[part,]
  test_Y = Y.scaled[-part,]
  
  # prepare the output
  train = list(X=train_X, Y=train_Y)
  test = list(X=test_X, Y=test_Y)
  
  # return the scaled and split partitions in a list
  out = list(train=train, test=test)
  
  return(out)
}
