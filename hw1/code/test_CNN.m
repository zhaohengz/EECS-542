function [ accuracy ] = test_CNN(model, input, label, batch_size)
% test the trained model

hit = 0;
[~, ~, ~, num] = size(input);
for i = 1 : num / batch_size
    [output,~] = inference(model, input(:,:,:, (i-1) * batch_size + 1 : i * batch_size));
    [~, I] = max(output);
    I = reshape(I, [batch_size 1]);
    hit = hit + sum(I == label((i-1) * batch_size + 1 : i * batch_size));
end

accuracy = hit / num;

end

