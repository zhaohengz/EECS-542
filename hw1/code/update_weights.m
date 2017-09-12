function updated_model = update_weights(model,grad,hyper_params)

num_layers = length(grad);
a = hyper_params.learning_rate;
lmda = hyper_params.weight_decay;
updated_model = model;

% TODO: Update the weights of each layer in your model based on the calculated gradients
