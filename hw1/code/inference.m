function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

% TODO: FORWARD PROPAGATION CODE

output = activations{end};
