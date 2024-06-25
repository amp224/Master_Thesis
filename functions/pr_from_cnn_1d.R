pr_from_cnn_1d = function(train, test, cnn_model, activation, q_taylor){
  # train: train partition of the dataset
  # test: test partition of the dataset
  # cnn_model: trained full CNN  model
  # activation: activation function (defined as function in R)
  # q_taylor: degree to which the Taylor expansion will be carried out
  
  if (activation=='tanh'){
    fun = tanh
  }
  else if (activation=='softplus'){
    fun = softplus
  }
  
  # extract the weights from the model
  w = as.vector(get_weights(cnn_model)[[1]])
  # extract the bias
  b = as.double(get_weights(cnn_model)[[2]])
  
  # number of variables
  p = dim(train$X)[2]
  
  # number of signals in the testing set
  n_test = length(test$Y)
  
  # kernel size
  k = length(w)
  
  # compute the vector with the Taylor coefficients
  g = taylor(fun, 0, order=q_taylor)$terms[,2]
  
  # obtain polynomial coefficients
  poly = coefs_from_1d_kernel(p, w, b, g)
  
  # make the predictions of the PR on the testing set
  PR.prediction.feature = array(dim=c(n_test, p-k+1))
  
  for (signal in 1:n_test){
    PR.prediction.feature[signal,] = evaluate_poly_1d(test$X[signal,], poly)
  }
  
  # input shape for the convolutional layer
  input_shape = c(length(train$X[1,]), 1)
  
  # create convolutional layer to later predict
  conv_layer = keras_model_sequential(
    layer_conv_1d(filters=1, kernel_size=k, activation=activation, 
                  input_shape=input_shape)
  )
  
  # set the weights of the new Conv layer equal to those of the trained layer
  # in the CNN
  set_weights(conv_layer, get_weights(cnn_model)[c(1,2)])
  
  # obtain predictions of the convolutional layer
  NN.prediction.feature = predict(conv_layer, test$X, verbose=0)
  # reshape to get rid of the channel dimension (only 1 channel)
  NN.prediction.feature = array_reshape(NN.prediction.feature, c(n_test, p-k+1))
  
  # Element-wise squared error between PR and NN feature maps
  MSE.NN.vs.PR.feature = apply((NN.prediction.feature-PR.prediction.feature)^2/n_test, 2, sum)
  
  # predictions of the CNN for the target
  NN.prediction.Y = predict(cnn_model, test$X, verbose=0)
  
  # the predictions of the PR for the target are simply a matrix product with 
  # the weights of the dense layer and adding its bias
  PR.prediction.Y = PR.prediction.feature %*% as.vector(get_weights(cnn_model)[[3]]) + as.double(get_weights(cnn_model)[[4]])
  
  # mean squared error between PR and CNN target
  MSE.NN.vs.PR.Y = sum((NN.prediction.Y-PR.prediction.Y)^2/n_test)
  
  # save everything in a list
  out = vector(mode='list')
  out[[1]] = train
  out[[2]] = test
  out[[3]] = g
  out[[4]] = cnn_model
  out[[5]] = poly
  out[[6]] = NN.prediction.feature
  out[[7]] = PR.prediction.feature
  out[[8]] = MSE.NN.vs.PR.feature
  out[[9]] = NN.prediction.Y
  out[[10]] = PR.prediction.Y
  out[[11]] = MSE.NN.vs.PR.Y
  
  
  names(out) = c("train", "test", "g", "cnn", "poly", 
                 "NN.prediction.feature", "PR.prediction.feature",
                 "MSE.NN.vs.PR.feature",
                 "NN.prediction.Y", "PR.prediction.Y",
                 "MSE.NN.vs.PR.Y"
                 )
  
  return(out)
}
