function [net_update, loss] = back_propagation(net, ground_truth, reg)
% @Des: Function uses to compute backward for network
% @Param:
%   net: Structure stores parameter of network
%   label: Ground truth label, format one-hot vector
% @Ret:
%   net_update: Structure stores gradient of parameter
% Structure of net_update {
%       net.dx: Gradient with input x
%       net.dw: Gradient with weights
%       net.db: Gradient with bias
% }

% [loss, dL] = svm_loss(net.layer{net.layer_num, 1}, ground_truth);

[loss, dL] = cross_entropy(net.layer{net.layer_num, 1}, ground_truth);

% GAMMA = 0.5;
% ALPHA = 1;
% [loss, dL] = focal_loss(net.layer{net.layer_num, 1}, ground_truth, GAMMA, ALPHA);

% [loss, dL] = L2_loss(net.layer{net.layer_num, 1}, ground_truth);

dz = dL;

index_layer = net.layer_num;

% Compute separately gradient for output layer
net_update.dx{index_layer, 1} = dz * net.weight{index_layer,1}';
net_update.dw{index_layer, 1} = net.layer{index_layer - 1,1}' * dz;
net_update.db{index_layer, 1} = sum(dz, 1);

% Compute gradient for hidden layer
for index_layer = net.layer_num-1:-1:2
    dout = net_update.dx{index_layer + 1, 1};
    
    if strcmp(net.activation_func, 'sigmoid')
    % Gradient with sigmoid
        dz = net.layer{index_layer,1} .* (1 - net.layer{index_layer,1});
        dz = dz .* dout;
    elseif strcmp(net.activation_func, 'relu')
        % Gradient with ReLU
        dz = dout;
        dz(net.layer{index_layer,1} <= 0) = 0;
    else
        disp('Not support activation function !!!');
    end

    % Gradient with input, weight ans bias
    net_update.dx{index_layer, 1} = dz * net.weight{index_layer,1}';
    net_update.dw{index_layer, 1} = net.layer{index_layer - 1,1}' * dz;
    net_update.db{index_layer, 1} = sum(dz, 1);
end

% Add L2 regularization
for index_layer = net.layer_num:-1:2
    loss = loss + 0.5*reg*sum(sum(net.weight{index_layer,1}.^2));
end

for index_layer = net.layer_num:-1:2
    net_update.dw{index_layer, 1} = net_update.dw{index_layer, 1} + ...
        reg * net.weight{index_layer, 1};
end
end