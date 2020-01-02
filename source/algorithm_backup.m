%% your classifer traing code here
function [model] = algorithm(x_train, y_train, x_valid, y_valid)

%% data & neuron number setting
num_neuron_input      = size(x_train, 2);
num_neuron_output     = 10;
num_data_train        = size(x_train, 1);

%% Parameters here %%
% num_neuron_hidden = [64;32]; % hidden neuron num
num_neuron_hidden = [64;32;16]; % hidden neuron num
% num_neuron_hidden = [100;64;32]; % hidden neuron num

% Weight initialization setting
init.weight_std         = 1e-1; % stdev of weight paramters
init.bias_std           = 1e-1;   % stdev of bias paramters
init.BATCH_SIZE         = 1;

% Training setting
training.N_EPOCH                    = 30;       % Num of epochs
training.REGULARIZE                 = 0;        % Regularization parameter
% training.REGULARIZE                 = 1e-4;        % Regularization parameter
training.BATCH_SIZE                 = init.BATCH_SIZE;
training.N_EPOCH_UNDERSAMPLING      = [0 20];

training.rule           = 'sgd';
config.learning_rate    = 5e-4;
% training.rule           = 'sgd_momentum';
% config.learning_rate    = 5e-3;
% config.momentum         = 0.9;

% training.save_period    = 250;    % Save period (unit: num batchsize)
% training.test_period    = 1250;    % Test period (should be multiple of save period)

training.save_period    = 1000;    % Save period (unit: num batchsize)
training.test_period    = 5000;    % Test period (should be multiple of save period)

%% Iinitialize Network %%
num_neuron            = [num_neuron_input; num_neuron_hidden; num_neuron_output];
[net, config]         = initialize_network(num_neuron, init, config);

%% Initializations Log for Training %%
training.num_input_graph = training.N_EPOCH * ...
    floor(num_data_train /(training.test_period*training.BATCH_SIZE));
training.graph_train = zeros(training.num_input_graph, 1);
training.graph_val = zeros(training.num_input_graph, 1);
training.loss = zeros(training.num_input_graph, 1);
training.index_graph = 0;
training.val_per_epoch = floor(num_data_train / training.test_period);

training_loss = 0;

%% Hold original data
x_train_orig = x_train;
y_train_orig = y_train;

% x_mean = sum(x_train_orig,1)/size(x_train_orig,1);
% x_train_orig = x_train_orig - repmat(x_mean, size(x_train_orig,1),1);
% x_valid = x_valid - repmat(x_mean, size(x_valid, 1), 1);

%% Training %%
for epoch = 1 : training.N_EPOCH
    % Learning rate decay 
    if epoch == 20
        config.learning_rate = config.learning_rate/10;
%         for layer_index = 2 : net.layer_num
%             net.weight{layer_index, 1} = net.best_weight{layer_index, 1};
%             net.bias{layer_index, 1} = net.best_bias{layer_index, 1};
%         end
    end
    
    % Check to use under-sampling
    if epoch <= training.N_EPOCH_UNDERSAMPLING(1)
        disp(['Epoch ',num2str(epoch),': Training in undersampling mode...']);
        [x_train, y_train] = preprocess_data(x_train_orig, y_train_orig);
        num_data_train = size(x_train, 1);
    elseif epoch > training.N_EPOCH_UNDERSAMPLING(1) && epoch <= training.N_EPOCH_UNDERSAMPLING(2)
        disp(['Epoch ',num2str(epoch),': Training in normal mode...']);
        if epoch == training.N_EPOCH_UNDERSAMPLING(1)+1
            for layer_index = 2 : net.layer_num
                net.weight{layer_index, 1} = net.best_weight{layer_index, 1};
                net.bias{layer_index, 1} = net.best_bias{layer_index, 1};
            end
        end
        x_train = x_train_orig;
        y_train = y_train_orig;
        num_data_train = size(x_train, 1);
    else
        [x_train, y_train] = preprocess_data(x_train_orig, y_train_orig);
%         disp(['Epoch ',num2str(epoch),': Training in attention mode...']);
%         x_train = x_train_orig(y_train_orig == 1 | y_train_orig == 7 | y_train_orig == 9 | y_train_orig == 10,:);
%         y_train = y_train_orig(y_train_orig == 1 | y_train_orig == 7 | y_train_orig == 9 | y_train_orig == 10,:);
        num_data_train = size(x_train, 1);
    end
        
    n_iter_per_epoch = num_data_train/training.BATCH_SIZE;
    order_index_train = randperm(num_data_train);
    for batch_idx = 1 : n_iter_per_epoch
        data_input = x_train(order_index_train((batch_idx-1)*...
            training.BATCH_SIZE+1:batch_idx*training.BATCH_SIZE),:);
        label = y_train(order_index_train((batch_idx-1)*...
            training.BATCH_SIZE+1:batch_idx*training.BATCH_SIZE),:);
        %% Foward computations
        [net, ~] = feed_forward(net, 'train', data_input);
        %% Backward computations
        [net_update, loss] = back_propagation(net, label, training.REGULARIZE);

        training_loss = training_loss + sum(loss);   % Used to monitor loss
        %% Weight update
        [net, config] = weight_update(net, net_update, training, config);

        % Save best model
        if mod(batch_idx, training.save_period) == 0 || ...
            batch_idx == n_iter_per_epoch
            train_acc = validation(net, 'train', x_train, y_train);
            val_acc = validation(net, 'train', x_valid, y_valid);
            
            if val_acc > net.best_val_acc
                for layer_index = 2 : net.layer_num
                  net.best_weight{layer_index, 1} = net.weight{layer_index, 1};
                  net.best_bias{layer_index, 1} = net.bias{layer_index, 1};
                end
                net.best_val_acc = val_acc;
                net.best_epoch = epoch;
            end
            if mod(batch_idx, training.test_period) == 0 || ... 
                batch_idx == n_iter_per_epoch % Test period
            
                training.index_graph = training.index_graph + 1;
                training.graph_train(training.index_graph) = 1 - train_acc/100;
                training.graph_val(training.index_graph) = 1 - val_acc/100;
                if epoch <= training.N_EPOCH_UNDERSAMPLING(1)
                    training.loss(training.index_graph) = training_loss/5000;
                elseif epoch > training.N_EPOCH_UNDERSAMPLING(1) && epoch <= training.N_EPOCH_UNDERSAMPLING(2)
                    if batch_idx == n_iter_per_epoch
                        training.loss(training.index_graph) = ...
                            training_loss/(mod(batch_idx,training.test_period)*training.BATCH_SIZE);
                    else
                        training.loss(training.index_graph) = ...
                            training_loss/((training.test_period + 1)*training.BATCH_SIZE);
                    end
                else
                    training.loss(training.index_graph) = training_loss/5000;
                end
                training_loss = 0;
                % command text
                performance_test = ['epoch: ', num2str(epoch), '  iteration: ', num2str(batch_idx), '  Error: ', num2str(1 - val_acc/100)];
                disp(performance_test);
                live_plot(training);
            end
        end
    end
end
model = net;

end