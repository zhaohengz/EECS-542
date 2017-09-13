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
x_hat = zeros(size(output));
% TODO: FORWARD CODE
tmp = permute(input, [1 2 4 3]);
mean_x = mean(reshape(tmp, [], num_channels), 1);
variance_x = var(reshape(tmp, [], num_channels), 1, 1);
for i = 1:hyper_params.filter_depth
    x_hat(:, :, i, :) = (input(:, :, i, :) - mean_x(i)) ./ sqrt(variance_x(i) + ep);
    output(:, :, i, :) = x_hat(:, :, i, :) .* params.W(i) + params.b(i);
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
    dv_input = zeros(size(input));
    grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
    dv_xhat = zeros(size(input));
    dv_var = zeros(size(variance_x));
    dv_mean = zeros(size(mean_x));
    % TODO: BACKPROP CODE
    for i = 1 : hyper_params.filter_depth
        grad.W(i) = sum(reshape(dv_output(:, :, i, :) .* x_hat(:, :, i, :), [], 1));
        grad.b(i) = sum(reshape(dv_output(:, :, i, :), [], 1));
        dv_xhat(:, :, i, :) = dv_output(:, :, i, :) * params.W(i);
        dv_var(i) = sum(reshape(dv_xhat(:, :, i, :) .* (input(:, :, i, :) - mean_x(i)), [], 1)) * -0.5 * (variance_x(i) + ep)^(-1.5);
        dv_mean(i) = sum(reshape(dv_xhat(:, :, i, :) * (-1.0) / sqrt(variance_x(i) + ep), [], 1))  + ...
            dv_var(i) * sum(reshape(-2 * (input(:, :, i, :) - mean_x(i)), [], 1)) / (out_height * out_width * batch_size);
        dv_input(:, :, i, :) = dv_xhat(:, :, i, :) / sqrt(variance_x(i) + ep) + ...
             dv_var(i) * 2 * (input(:, :, i, :) - mean_x(i)) / (out_height * out_width * batch_size) + ...
             dv_mean(i) / (out_height * out_width * batch_size);
    end
end
