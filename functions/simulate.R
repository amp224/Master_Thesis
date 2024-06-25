simulate = function(sample_size, signal_size, range, 
                    case, kernel_size, epochs,
                    activation, q_taylor){

  # sample_size: number of signals in the dataset to be generated
  # signal_size: either length of the 1D signal or size of the square image
  # range: range for the generation of l.c. coefficients for the target Y
  # case: string specifying hether 1D or 2D signals are considered
  # kernel_size: size of the 1D kernel or of the square 2D kernel
  # epochs: number of epochs the CNN should be trained
  # activation: string specifying the activation function ('tanh' or 'softplus')
  # q_taylor: Taylor order to which the PR will be expanded
  
  # generate the data
  data = generate_data(case, sample_size, signal_size, 
                       kernel_size, c(-range,range))
  
  # create the train-test partition
  split = scale_and_split(case, data, split_prop=0.8)
  
  if (case=='1D'){
    input_shape = c(signal_size, 1)
    
    # CNN to be trained
    cnn_model <- keras_model_sequential() %>%
      layer_conv_1d(filters=1, kernel_size=kernel_size, activation=activation, input_shape=input_shape) %>%
      layer_flatten() %>%
      layer_dense(units=1, activation = "linear")

    
    # compile the CNN before training
    cnn_model %>% compile(
      loss="mean_squared_error",
      optimizer=optimizer_adam()
    )
    
    # train the CNN
    cnn_history = cnn_model %>% fit(
      split$train$X, split$train$Y,
      view_metrics=F, verbose=0,
      epochs=epochs
    )
    

    out = pr_from_cnn_1d(split$train, split$test, cnn_model, 'tanh', q_taylor)
    
  }
  
  else if (case=='2D'){
    input_shape = c(signal_size, signal_size, 1)
    
    # CNN to be tranined
    cnn_model <- keras_model_sequential() %>%
      layer_conv_2d(filters=1, kernel_size=kernel_size, activation=activation, input_shape=input_shape) %>%
      layer_flatten() %>%
      layer_dense(units=1, activation = "linear")
    
    # compile the CNN before training
    cnn_model %>% compile(
      loss="mean_squared_error",
      optimizer=optimizer_adam()
    )
    
    # train the CNN on the data
    cnn_history = cnn_model %>% fit(
      split$train$X, split$train$Y,
      view_metrics=F, verbose=0,
      epochs=epochs
    )
    
    out = pr_from_cnn_2d(split$train, split$test, cnn_model, activation, q_taylor)
  }
  
  else{
    message(paste0("The case ", case, " has not been implemented."))
  }
  
  return (out)
}
