function [x_test] = createDatasetTest(test_feat)
%% Arguments %%
% train_feat : name of train data
% test_feat : name of test data
% x_test : test data
% y_test : testlabel
%% Codes %%
filename = test_feat;
x_test = csvread(filename);
x_test = x_test';
% filename = test_label;
% y_test = csvread(filename);