function [net, config_res] = weight_update(net, net_update, training, config)
% @Des: Function uses for updating parameter of network
% $Param:
%   net: Stores paramaters of network
%   net_update: Store gradient of parameters of network
%   l_rate: Learning rate
% @Ret:
%   net: Network's parameters is updated

config_res = config;

if strcmp(training.rule, 'sgd')
    l_rate = config_res.learning_rate;
    for index_layer = 2 : net.layer_num
        net.weight{index_layer, 1} = net.weight{index_layer, 1} - l_rate * (1/training.BATCH_SIZE) * net_update.dw{index_layer, 1};
        net.bias{index_layer, 1} = net.bias{index_layer, 1} - l_rate * (1/training.BATCH_SIZE) * net_update.db{index_layer, 1};
    end
elseif strcmp(training.rule, 'sgd_momentum')
    v_w = config_res.velocity_weight;
    v_b = config_res.velocity_bias;
    beta = config_res.momentum;
    alpha = config_res.learning_rate;
    for index_layer = 2 : net.layer_num
        v_w{index_layer, 1} = beta*v_w{index_layer, 1} - alpha*net_update.dw{index_layer, 1};
        v_b{index_layer, 1} = beta*v_b{index_layer, 1} - alpha*net_update.db{index_layer, 1};
        net.weight{index_layer, 1} = net.weight{index_layer, 1} + v_w{index_layer, 1};
        net.bias{index_layer, 1} = net.bias{index_layer, 1} + v_b{index_layer, 1};
    end
    config_res.velocity_weight = v_w;
    config_res.velocity_bias = v_b;
end 
end