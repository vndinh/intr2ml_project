%% your classifer traing code here
function [model] = algorithm(x_train, y_train, x_valid, y_valid)

%% data & neuron number setting
num_neuron_input      = size(x_train, 2);
num_neuron_output     = 10;
num_data_train        = size(x_train, 1);

%% Parameters here %%
num_neuron_hidden = [20]; % hidden neuron num
% num_neuron_hidden = [32;16]; % hidden neuron num
% num_neuron_hidden = [64;32;16]; % hidden neuron num

% Weight initialization setting
init.weight_std         = 1e-1; % stdev of weight paramters
init.bias_std           = 1e-1;   % stdev of bias paramters
init.BATCH_SIZE         = 1;

% Training setting
training.N_EPOCH                    = 50;       % Num of epochs
% training.REGULARIZE                 = 0;        % Regularization parameter
training.REGULARIZE                 = 1e-5;        % Regularization parameter
training.BATCH_SIZE                 = init.BATCH_SIZE;
training.MODE                       = 'train';
% training.MODE                       = 'finetune';

training.rule           = 'sgd';
config.learning_rate    = 1e-3;
% training.rule           = 'sgd_momentum';
% config.learning_rate    = 5e-3;
% config.momentum         = 0.9;

% training.save_period    = 125;    % Save period (unit: num batchsize)
% training.test_period    = 625;    % Test period (should be multiple of save period)

training.save_period    = 500;    % Save period (unit: num batchsize)
training.test_period    = 2500;    % Test period (should be multiple of save period)

%% Iinitialize Network %%
num_neuron            = [num_neuron_input; num_neuron_hidden; num_neuron_output];
if strcmp(training.MODE, 'train')
    disp('Training mode...');
    [net, config]         = initialize_network(num_neuron, init, config);
elseif strcmp(training.MODE, 'finetune')
    model_path = './ckpt/model_ckpt_2018-06-09_18_25_53.mat';
    disp(['Finetune from ', model_path]);
    load(model_path);
    net = model;
end

%% Initializations Log for Training %%
training.num_input_graph = training.N_EPOCH * floor(num_data_train /(training.test_period*training.BATCH_SIZE));
training.graph_train    = zeros(training.num_input_graph, 1);
training.graph_val      = zeros(training.num_input_graph, 1);
training.loss           = zeros(training.num_input_graph, 1);
training.index_graph    = 0;
training.val_per_epoch  = floor(num_data_train / training.test_period);
training_loss           = 0;

%% Hold original data
x_train_orig = x_train;
y_train_orig = y_train;
x_valid_orig = x_valid;
y_valid_orig = y_valid;

x_dataset = [x_train; x_valid];
y_dataset = [y_train; y_valid];

%% Configuration for K-folds
k_folds = 10;
[x_dataset, y_dataset, start_id, end_id] = make_folds(x_dataset, y_dataset, k_folds);
idx_folds = 1;

training.avg_valid_folds     = zeros(1, k_folds);
training.id_avg_valid_folds = 1;
training.valid_fold     = zeros(1, ceil(1948*9/training.test_period));
training.id_valid_fold = 1;

%% Training %%
for epoch = 1 : training.N_EPOCH
    %% Prepare data for epoch
    x_train = [];
    y_train = [];
    x_valid = x_dataset(start_id(idx_folds):end_id(idx_folds),:);
    y_valid = y_dataset(start_id(idx_folds):end_id(idx_folds),:);
    for k = 1:k_folds
        if k ~= idx_folds
            x_train = [x_train; x_dataset(start_id(k):end_id(k),:)];
            y_train = [y_train; y_dataset(start_id(k):end_id(k),:)];
        end
    end
    
    num_data_train = size(x_train, 1);
    n_iter_per_epoch = num_data_train/training.BATCH_SIZE;
    order_index_train = randperm(num_data_train);
    
    % Learning rate decay 
    if epoch == 20
        config.learning_rate = config.learning_rate/10;
    elseif epoch == 40
        config.learning_rate = config.learning_rate/10;
    end
    
    for batch_idx = 1 : n_iter_per_epoch
        % Get batch data
        data_input = x_train(order_index_train((batch_idx-1)*...
            training.BATCH_SIZE+1:batch_idx*training.BATCH_SIZE),:);
        label = y_train(order_index_train((batch_idx-1)*...
            training.BATCH_SIZE+1:batch_idx*training.BATCH_SIZE),:);
        
        % Foward computations
        [net, ~] = feed_forward(net, 'train', data_input);
        % Backward computations
        [net_update, loss] = back_propagation(net, label, training.REGULARIZE);

        % Weight update
        [net, config] = weight_update(net, net_update, training, config);

        % Log and save best model
        training_loss = training_loss + sum(loss);   % Used to monitor loss
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

                if batch_idx == n_iter_per_epoch
                    training.loss(training.index_graph) = ...
                        training_loss/(mod(batch_idx,training.test_period)*training.BATCH_SIZE);
                    training.valid_fold(training.id_valid_fold) = val_acc;
                    training.avg_valid_folds(training.id_avg_valid_folds)...
                        = sum(training.valid_fold)/ceil(1948*9/training.test_period);
                    training.id_avg_valid_folds = training.id_avg_valid_folds + 1;
                    training.valid_fold = zeros(1, ceil(1948*9/training.test_period));
                    training.id_valid_fold = 1;
                else
                    training.loss(training.index_graph) = ...
                        training_loss/((training.test_period + 1)*training.BATCH_SIZE);
                    training.valid_fold(training.id_valid_fold) = val_acc;
                    training.id_valid_fold = training.id_valid_fold + 1;
                end

                training_loss = 0;
                % command text
                performance_test = ['epoch: ', num2str(epoch), '  iteration: ', num2str(batch_idx), '  Error: ', num2str(1 - val_acc/100)];
                disp(performance_test);
                live_plot(training);
            end
        end
    end
    
    % Reset fold index
    idx_folds = idx_folds + 1;
    if idx_folds > k_folds
        idx_folds = 1;
        disp(['ACCURACY K-FOLDS: ', num2str(sum(training.avg_valid_folds)/k_folds)]);
        training.avg_valid_folds = zeros(1, k_folds);
    end
end
model = net;

end
