% ----------------------------------------------------------------------
% input: any n-d array
% output: same as input
% dv_output: same as input
% dv_input: same as input
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_relu(input, params, hyper_params, backprop, dv_output)
% Rectified linear unit activation function

output = zeros(size(input));
% TODO: FORWARD CODE

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
    dv_input = zeros(size(input));
    % TODO: BACKPROP CODE
end
