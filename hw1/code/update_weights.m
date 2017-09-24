function [updated_model, velocity] = update_weights(model,grad,last_v, hyper_params)

num_layers = length(grad);
a = hyper_params.learning_rate;
lmda = hyper_params.weight_decay;
p = hyper_params.momentum;
updated_model = model;

% TODO: Update the weights of each layer in your model based on the calculated gradients
velocity = cell(num_layers, 1);
temp = struct('W', [], 'b', []);
for i = 1 : num_layers
    temp.W = a * grad{i}.W + p * last_v{i}.W;
    temp.b = a * grad{i}.b + p * last_v{i}.b;
    velocity{i} = temp;
    updated_model.layers(i).params.W = updated_model.layers(i).params.W * (1 - a * lmda) - temp.W;
    updated_model.layers(i).params.b = updated_model.layers(i).params.b * (1 - a * lmda) - temp.b;
end
