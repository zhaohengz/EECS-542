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
max_idx = zeros(out_height, out_width, num_channels, batch_size, 2);
for i = 1 : out_height
    for j = 1 : out_width
        output(i, j, :, :) = max(max(input((i-1) * filter_size + 1 : i * filter_size, (j-1) *filter_size + 1 : j * filter_size, :, :)));
    end
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    for i = 1 : out_height
        for j = 1 : out_width
            for k = 1 : num_channels
                for l = 1 : batch_size
                    [x, y] = ind2sub(size(input((i-1) * filter_size + 1 : i * filter_size, (j-1) *filter_size + 1 : j * filter_size, k, l)), ...
                        find(input((i-1) * filter_size + 1 : i * filter_size, (j-1) *filter_size + 1 : j * filter_size, k, l) ==  output(i, j, k, l)));
                    dv_input((i-1) * filter_size + x, (j-1) *filter_size + y, k, l) = dv_output(i, j, k, l);
                end
            end
        end
    end
end
