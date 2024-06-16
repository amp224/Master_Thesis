pr_from_cnn_2d = function(train, test, cnn_model, activation, q_taylor=2){
  # train: train partition of the dataset
  # test: test partition of the dataset
  # cnn_model: trained 1-Convolutional layer model
  # activation: activation function (defined as function in R)
  # q_taylor: order of the Taylor expansion (only 2 for 2D)
  
  # extract the weights from the model
  w = get_weights(cnn_model)[[1]]
  k = dim(w)[1]
  
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
  g = taylor(activation, 0, order=q_taylor)$terms[,2]
  
  # obtain the polynomial coefficients
  poly = coefs_from_2d_kernel(k, dims, w, b, g)
  
  # make the predictions of the PR on the testing set
  PR.prediction = array(dim=c(n_test,dims-k+1))
  
  for (image in 1:n_test){
    PR.prediction[image,,] = evaluate_poly_2d(k, test$X[image,,], poly)
  }
  
  # obtain predictions of the convolutional layer
  NN.prediction = predict(cnn_model, test$X)
  # reshape to get rid of the channel dimension
  NN.prediction = array_reshape(NN.prediction, dim=c(n_test, dims-k+1))
  
  # element-wise squared error between PR and NN predictions
  SE.NN.vs.PR = (PR.prediction - NN.prediction)^2
  
  out = vector(mode='list')
  out[[1]] = train
  out[[2]] = test
  out[[3]] = g
  out[[4]] = cnn_model
  out[[5]] = poly
  out[[6]] = NN.prediction
  out[[7]] = PR.prediction
  out[[8]] = SE.NN.vs.PR
  
  names(out) = c("train", "test", "g", "cnn", "poly", "NN.prediction",
                 "PR.prediction", "SE.NN.vs.PR")
  
  return(out)
}
