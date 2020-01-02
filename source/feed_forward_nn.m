function [net, pred] = feed_forward(net, mode, data_input)
% @Des: Utility function used to feedforward input to network
% @Param:
%   data_input: An input image
%   net: Structure stores parameter of network
% @Ret:
%   net: Return structure of network
%   pred: Output prediction

net.layer{1,1} = data_input;
batch_size = size(data_input,1);

pred = zeros(batch_size, 1);
for id_batch = 1:batch_size
    if strcmp(mode, 'train')
        for index_layer = 2 : net.layer_num
          z = net.layer{index_layer-1, 1}(id_batch, :) * ...
              net.weight{index_layer, 1} + net.bias{index_layer, 1};

          % If hidden layer, activation is sigmoid, else using softmax
          if index_layer < net.layer_num
              if strcmp(net.activation_func, 'sigmoid')
                  net.layer{index_layer, 1}(id_batch, :) = sigmoid(z);
              elseif strcmp(net.activation_func, 'relu')
                  net.layer{index_layer, 1}(id_batch, :) = relu(z);
              else
                  disp('Not support activation function !!!');
              end
          else
              net.layer{index_layer, 1}(id_batch, :) = z;
          end
        end
        [~, pred(id_batch)] = max(net.layer{net.layer_num, 1}(id_batch,:));
    elseif strcmp(mode, 'pred')
        for index_layer = 2 : net.layer_num
          z = net.layer{index_layer-1, 1}(id_batch, :) * ...
              net.best_weight{index_layer, 1} + net.best_bias{index_layer, 1};

          % If hidden layer, activation is sigmoid, else using softmax
          if index_layer < net.layer_num
              if strcmp(net.activation_func, 'sigmoid')
                  net.layer{index_layer, 1}(id_batch, :) = sigmoid(z);
              elseif strcmp(net.activation_func, 'relu')
                  net.layer{index_layer, 1}(id_batch, :) = relu(z);
              else
                  disp('Not support activation function !!!');
              end
          else
            net.layer{index_layer, 1}(id_batch, :) = z;
          end
        end
        [~, pred(id_batch)] = max(net.layer{net.layer_num, 1}(id_batch, :));    
    end
end
end