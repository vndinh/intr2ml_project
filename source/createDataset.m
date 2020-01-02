function [x_train,y_train,x_valid,y_valid] = createDataset(train_feat, train_label,valid_feat, valid_label)
%% Arguments %%
% train_feat : name of train data
% test_feat : name of test data
% x_train : train data
% y_train : train label
% x_valid : validation data
% y_valid : validation label
%% Codes %%

filename = train_feat;
x_train = csvread(filename);
x_train = x_train';
filename = train_label;
y_train = csvread(filename);

filename = valid_feat;
x_valid = csvread(filename);
x_valid = x_valid';
filename = valid_label;
y_valid = csvread(filename);

%filename = 'test_feat.csv';
%x_valid = csvread(filename);
%filename = 'test_label.csv';
%y_valid = csvread(filename);