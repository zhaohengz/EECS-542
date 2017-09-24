% Basic script to create a new network model
load_MNIST_data;
addpath layers;

l = [init_layer('conv',struct('filter_size',5,'filter_depth',1,'num_filters',6))
	init_layer('pool',struct('filter_size',2,'stride',2))
	init_layer('leaky_relu',[])
    init_layer('conv', struct('filter_size',5, 'filter_depth', 6, 'num_filters', 16))
    init_layer('pool',struct('filter_size',2,'stride',2))
	init_layer('leaky_relu',[])
    init_layer('conv', struct('filter_size',3, 'filter_depth', 16, 'num_filters', 120))
	init_layer('leaky_relu',[])
	init_layer('flatten',struct('num_dims',4))
    init_layer('linear',struct('num_in',480,'num_out',10))
	init_layer('softmax',[])];

model = init_model(l,[28 28 1],10,true);

% Example calls you might make:
numIters = 50;
[model, loss] = train(model,struct('train', train_data, 'test', test_data)...
    , struct('train', train_label, 'test', test_label)...
    , struct('learning_rate', 0.03, 'weight_decay', .0005, 'batch_size', 1000),numIters);
%trained_model = load('save_file.mat');

% fprintf('Accuracy: %.2f%% \n',accuracy * 100);
