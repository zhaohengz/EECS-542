% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% labels: batch_size x 1
% ----------------------------------------------------------------------

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));

% TODO: CALCULATE LOSS
[N, M] = size(input);
temp = zeros(size(input));
input_log = -log(input);
for i = 1 : M
    temp(labels(i), i) = 1;
end
input_log = sum(temp .* input_log);
loss = sum(input_log) / M;

dv_input = ones(size(input));
if backprop
    % TODO: BACKPROP CODE
    dv_input = temp .* dv_input ./ input / -M;
end

