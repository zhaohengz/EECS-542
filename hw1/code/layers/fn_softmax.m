% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);
output = zeros(num_classes, batch_size);
% TODO: FORWARD CODE
exp_x = exp(input);
output = exp_x ./ (ones(num_classes, 1) * sum(exp_x, 1));

dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
	dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    for i = 1 : num_classes
        for j = 1 : num_classes
            temp(i,j,:) = output(i,:) .* ((i==j) - output(j, :));
        end
    end
    for i = 1 : batch_size
       dv_input(:, i) = temp(:,:,i) * dv_output(:,i);
    end
end
