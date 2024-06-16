pr_from_cnn_1d = function(train, test, cnn_model, activation, q_taylor){
  # train: train partition of the dataset
  # test: test partition of the dataset
  # cnn_model: trained 1-Convolutional layer  model
  # activation: activation function (defined as function in R)
  # q_taylor: degree to which the Taylor expansion will be carried out
  

    
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
  g = taylor(activation, 0, order=q_taylor)$terms[,2]
  
  # obtain polynomial coefficients
  poly = coefs_from_1d_kernel(p, w, b, g)
  
  # make the predictions of the PR on the testing set
  PR.prediction = array(dim=c(n_test, p-k+1))
  
  for (signal in 1:n_test){
    PR.prediction[signal,] = evaluate_poly_1d(test$X[signal,], poly)
  }
  
  # obtain predictions of the convolutional layer
  NN.prediction = predict(cnn_model, test$X)
  # reshape to get rid of the channel dimension (only 1 channel)
  NN.prediction = array_reshape(NN.prediction, c(n_test, p-k+1))
  
  # Element-wise squared error between PR and NN
  SE.NN.vs.PR = (NN.prediction-PR.prediction)^2
  
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
