% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (filter_depth)
% params.W: filter_depth x 1
% params.b: filter_depth x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_bn(input, params, hyper_params, backprop, dv_output)
ep = 1e-5; % for stability
[out_height,out_width,num_channels,batch_size] = size(input);
assert(hyper_params.filter_depth == num_channels, 'Filter depth does not match number of input channels');
output = zeros(out_height,out_width,num_channels,batch_size);
% TODO: FORWARD CODE

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
    dv_input = zeros(size(input));
    grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
    % TODO: BACKPROP CODE

end
