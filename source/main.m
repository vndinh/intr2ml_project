close all;
clear; clc;
%% algorithm %%
[x_train,y_train,x_valid,y_valid] = createDataset('train_feat.csv', 'train_label.csv','valid_feat.csv', 'valid_label.csv');

%% training 
model = algorithm(x_train, y_train, x_valid, y_valid);

%% test 
x_test       = createDatasetTest('test_feat.csv');
test_pred    = validation(model, 'pred', x_test);
export_prediction(test_pred, 'test_pred.csv');

%% Analysis here 
pred_train = validation(model, 'pred', x_train);
pred_valid = validation(model, 'pred', x_valid);

plot_confusion_matrix(y_train, pred_train, 'Training');
plot_confusion_matrix(y_valid, pred_valid, 'Validation');


%% Find Accuracy
x_dataset = [x_train; x_valid];
y_dataset = [y_train; y_valid];
train_acc = validation(model, 'pred', x_train, y_train);
valid_acc = validation(model, 'pred', x_valid, y_valid);
dataset_acc = validation(model, 'pred', x_dataset, y_dataset);
fprintf('Training accuracy: %.2f %% \n', train_acc);
fprintf('Validation accuracy: %.2f %% \n', valid_acc);
fprintf('Dataset accuracy: %.2f %% \n', dataset_acc);
fprintf('Best epoch: %d \n', model.best_epoch);

%% Save model
path = './ckpt/';
save_model(model, path, 'accuracy_log.txt', train_acc, valid_acc);
