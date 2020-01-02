function [net, config_res] = initialize_network(num_neuron, init, config)
% Structure of net {
%       net.layer_num: Number of layers
%       net.num_neron: Containing number of neurons of each layer
%       net.layer: Outputs of each layer
%       net.weight: Weigts of each layer
%       net.bias: Bias of each layer except input
% }

config_res = config;

%% Initialize structure
net.layer_num = length(num_neuron);
net.num_neuron = num_neuron;
% net.activation_func = 'sigmoid';
net.activation_func = 'relu';

%% Initialize each layer
net.layer = cell(net.layer_num,1);
for layer_index = 1 : net.layer_num
      net.layer{layer_index, 1} = zeros(init.BATCH_SIZE, net.num_neuron(layer_index, 1)); 
end

%% Initialize weight
net.weight = cell(net.layer_num, 1);
net.best_weight = cell(net.layer_num, 1);
config_res.velocity_weight = cell(net.layer_num, 1);
for layer_index = 2 : net.layer_num
      net.weight{layer_index, 1} = init.weight_std * randn(net.num_neuron(layer_index-1, 1), net.num_neuron(layer_index, 1));
      net.best_weight{layer_index, 1} = net.weight{layer_index, 1};
      config_res.velocity_weight{layer_index, 1} = zeros(net.num_neuron(layer_index-1, 1), net.num_neuron(layer_index, 1));
end
%% Initialize bias
net.bias = cell(net.layer_num, 1);
net.best_bias = cell(net.layer_num, 1);
config_res.velocity_bias = cell(net.layer_num, 1);
for layer_index = 2 : net.layer_num
      net.bias{layer_index, 1} = init.bias_std * randn(1, net.num_neuron(layer_index, 1));
      net.best_bias{layer_index, 1} = net.bias{layer_index, 1};
      config_res.velocity_bias{layer_index, 1} = zeros(1, net.num_neuron(layer_index, 1));
end

%% Initialize save loss
net.best_val_acc = 0;
net.best_epoch = 0;
end