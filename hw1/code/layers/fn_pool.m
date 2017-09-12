% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, filter_size)
% dv_output: same as output
% dv_input: same as input
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_pool(input, params, hyper_params, backprop, dv_output)
filter_size = hyper_params.filter_size;
stride = hyper_params.stride;
[~,~,num_channels,batch_size] = size(input);
assert(mod(size(input,1) - filter_size, stride)==0,...
	sprintf('Unsuitable stride and filter size'));
out_height = (size(input,1) - filter_size)/stride + 1;
out_width = (size(input,2) - filter_size)/stride + 1;
output = zeros(out_height,out_width,num_channels,batch_size);
% TODO: FORWARD CODE
for i = 1 : out_height
    for j = 1 : out_width
        output[i, j, :, :] = max(input[(i-1) * filter_size + 1 : i * filter_size, (j-1) *filter_size + 1 : j * filter_size, :, :]);
    end
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
end
