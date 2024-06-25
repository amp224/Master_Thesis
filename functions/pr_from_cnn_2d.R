pr_from_cnn_2d = function(train, test, cnn_model, activation, q_taylor=2){
  # train: train partition of the dataset
  # test: test partition of the dataset
  # cnn_model: trained full CNN model
  # activation: string defining activation function ('tanh' or 'softplus')
  # q_taylor: order of the Taylor expansion (only 2 for 2D)
  
  # set the activation functions for Taylor
  if (activation=='tanh'){
    fun = tanh
  }
  else if (activation=='softplus'){
    fun = softplus
  }
  
  # extract the weights from the model
  w = get_weights(cnn_model)[[1]]
  k = dim(w)[1]
  
  # get weights to the right shape for later manipulation
  if (k==1){
    w = as.double(w)
  }
  else if (k==2){
    w = array_reshape(w, dim=c(2,2))
  }
  
  # extract the bias from the model
  b = as.double(get_weights(cnn_model)[[2]])
  
  # dimensions of the images of the dataset
  dims = dim(train$X[1,,])
  
  # number of images in the test partition
  n_test = length(test$Y)
  
  # compute the Taylor coefficients
  g = taylor(fun, 0, order=q_taylor)$terms[,2]
  
  # obtain the polynomial coefficients
  poly = coefs_from_2d_kernel(k, dims, w, b, g)
  
  # make the predictions of the PR on the testing set
  PR.prediction.feature = array(dim=c(n_test,dims-k+1))
  
  for (image in 1:n_test){
    PR.prediction.feature[image,,] = evaluate_poly_2d(k, test$X[image,,], poly)
  }
  
  # input shape for the convolutional layer
  input_shape = c(dims, 1)
  
  # create single convolutional layer
  conv_layer = keras_model_sequential(
    layer_conv_2d(filters=1, kernel_size=k, activation=activation, 
                  input_shape=input_shape)
  )
  
  # set weights of the conv layer equal to those of the trained conv layer
  # within the CNN
  set_weights(conv_layer, get_weights(cnn_model)[c(1,2)])
  
  # obtain predictions of the convolutional layer
  NN.prediction.feature = predict(conv_layer, test$X, verbose=0)
  # reshape to get rid of the channel dimension
  NN.prediction.feature = array_reshape(NN.prediction.feature, dim=c(n_test, dims-k+1))
  
  # element-wise squared error between PR and NN predictions of the feature maps
  MSE.NN.vs.PR.feature = apply((PR.prediction.feature - NN.prediction.feature)^2/n_test, c(2,3), sum)
  
  # CNN predictions for the target
  NN.prediction.Y = predict(cnn_model, test$X, verbose=0)
  
  # the PR predictions for the target are simply scalar products with the
  # weights of dense layer
  PR.prediction.Y = rep(0, n_test)
  
  for (i in 1:n_test){
    PR.prediction.Y[i] = as.vector(t(PR.prediction.feature[i,,])) %*% get_weights(cnn_model)[[3]] + as.double(get_weights(cnn_model)[[4]])
  }
  
  # mean squared error between the target predictions
  MSE.NN.vs.PR.Y = sum((NN.prediction.Y-PR.prediction.Y)^2/n_test)
  
  # return everything in a list
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
