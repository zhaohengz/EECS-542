function [model, loss] = train(model,input,label,params,numIters)

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% Learning rate
if isfield(params,'learning_rate') lr = params.learning_rate;
else lr = .01; end
% Weight decay
if isfield(params,'weight_decay') wd = params.weight_decay;
else wd = .0005; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 128; end

% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file= params.save_file;
else save_file = 'model.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd, 'momentum', 0.5);

[height, width, channels, num] = size(input.train);
num_layers = length(model.layers);
velocity = cell(num_layers, 1);
for i = 1 : num_layers
    velocity{i} = struct('W', 0, 'b', 0);
end
training_loss = [];
test_loss = [];
accuracy_list = [];

handle = datestr(datetime('now'));

log_file = sprintf('log/train-%s-%f-%f-%f.log', handle, update_params.learning_rate...
    , update_params.weight_decay, update_params.momentum);
fid = fopen(log_file, 'w');

for i = 1:numIters
	% TODO: Training code
    for j = 1 : num / batch_size
        batch = (j - 1) * batch_size + 1: j * batch_size;
        batch_input = input.train(:, :, :, batch);
        ground_truth = label.train(batch);
        [output, activations] = inference(model, batch_input);
        [loss, dv_input] = loss_crossentropy(output, ground_truth, [], true);
        training_loss = [training_loss, loss];
        [grad] = calc_gradient(model, batch_input, activations, dv_input);
        [model, velocity] = update_weights(model, grad, velocity, update_params);
        msg_1 = sprintf('Training loss after epoch %d batch %d: %f', i, j, loss);
        [output, activations] = inference(model, input.test);
        [loss, ~] = loss_crossentropy(output, label.test, [], false);
        test_loss = [test_loss, loss];
        msg_2 = sprintf('Test loss:%f', loss);
        if mod(j, 20) == 0
            fprintf('%s, %s\n', msg_1, msg_2);
            fprintf(fid, '%s, %s\n', msg_1, msg_2);
        end
    end
    update_params.momentum = 0.9;
    accuracy = test_CNN(model, input.test, label.test, 10000);
    accuracy_list = [accuracy_list accuracy];

    model_file = sprintf('models/train-%d-%s-%f-%f-%f.mat', i, handle, update_params.learning_rate...
    , update_params.weight_decay, update_params.momentum);
    save(model_file, 'model');

    loss_file = sprintf('loss/train-%d-%s-%f-%f-%f.mat', i, handle, update_params.learning_rate...
    , update_params.weight_decay, update_params.momentum); 
    save(loss_file, 'training_loss', 'test_loss', 'accuracy');

    fprintf('Accuracy after epoch %d: %f\n', i, accuracy);
    fprintf(fid, 'Accuracy after epoch %d: %f\n', i, accuracy);
end

fclose(fid);
loss = [train_loss, test_loss];
