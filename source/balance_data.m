function [x_train_b, y_train_b] = balance_data(x_train, y_train)
N_MINORITY = 500;
N_CLASS = 10;
idx_label = cell(N_CLASS, 1);
x_train_b = zeros(N_MINORITY*N_CLASS, size(x_train, 2));
y_train_b = zeros(N_MINORITY*N_CLASS, size(y_train, 2));
for i = 1:N_CLASS
    idx_label{i,1} = find(y_train==i);
    mask_select = randperm(size(idx_label{i,1},1));
    x_train_b((i-1)*N_MINORITY+1:i*N_MINORITY,:) = x_train(idx_label{i,1}(mask_select(1:N_MINORITY)),:);
    y_train_b((i-1)*N_MINORITY+1:i*N_MINORITY,:) = y_train(idx_label{i,1}(mask_select(1:N_MINORITY)),:);
end
idx_permute = randperm(N_MINORITY*N_CLASS);
x_train_b = x_train_b(idx_permute, :);
y_train_b = y_train_b(idx_permute, :);
end