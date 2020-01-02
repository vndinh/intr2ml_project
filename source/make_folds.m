function [x_data, y_data, start_id, end_id] = make_folds(x, y, k_folds)
N_CLASS = size(unique(y), 1);
N_DATA = size(x, 1);

start_id = zeros(1, k_folds);
end_id = zeros(1, k_folds);
x_data = zeros(size(x));
y_data = zeros(size(y));

remain_per_class = zeros(1, N_CLASS);
id_start_class = zeros(1, N_CLASS);
id_start_class_in_fold = zeros(1, N_CLASS);

for i = 1:N_CLASS
    remain_per_class(i) = sum(y == i);
end
n_data_per_fold_of_class = floor(remain_per_class/k_folds);

id_start_class(1) = 1;
id_start_class_in_fold(1) = 1;
for i = 2:N_CLASS
    id_start_class(i) = sum(remain_per_class(1:i-1))+1;
    id_start_class_in_fold(i) = sum(n_data_per_fold_of_class(1:i-1))+1;
end

n_per_fold = sum(n_data_per_fold_of_class);

%% Sorting data correspond to class
[y, idx] = sort(y);
x = x(idx,:);

cnt_last_fold = 1;
%% Create each fold has same distribution
for i = 1:k_folds
    for c = 1:N_CLASS
        if i < k_folds
            if c < N_CLASS
                x_data((i-1)*n_per_fold+id_start_class_in_fold(c): ...
                       (i-1)*n_per_fold+id_start_class_in_fold(c+1)-1,:) = ...
                    x(id_start_class(c)+(i-1)*n_data_per_fold_of_class(c): ...
                      id_start_class(c)+  i  *n_data_per_fold_of_class(c)-1,:);
                y_data((i-1)*n_per_fold+id_start_class_in_fold(c): ...
                       (i-1)*n_per_fold+id_start_class_in_fold(c+1)-1,:) = ...
                    y(id_start_class(c)+(i-1)*n_data_per_fold_of_class(c): ...
                      id_start_class(c)+  i  *n_data_per_fold_of_class(c)-1,:);
            else
                x_data((i-1)*n_per_fold+id_start_class_in_fold(c):i*n_per_fold,:) = ...
                    x(id_start_class(c)+(i-1)*n_data_per_fold_of_class(c): ...
                      id_start_class(c)+i*n_data_per_fold_of_class(c)-1, :);
                y_data((i-1)*n_per_fold+id_start_class_in_fold(c):i*n_per_fold,:) = ...
                    y(id_start_class(c)+(i-1)*n_data_per_fold_of_class(c): ...
                      id_start_class(c)+i*n_data_per_fold_of_class(c)-1, :);
            end
        else
            x_data((i-1)*n_per_fold+cnt_last_fold: ...
                   (i-1)*n_per_fold+cnt_last_fold+remain_per_class(c)-1,:) = ...
                 x(id_start_class(c)+(i-1)*n_data_per_fold_of_class(c): ...
                   id_start_class(c)+(i-1)*n_data_per_fold_of_class(c)+remain_per_class(c)-1, :);
            y_data((i-1)*n_per_fold+cnt_last_fold: ...
                   (i-1)*n_per_fold+cnt_last_fold+remain_per_class(c)-1,:) = ...
                 y(id_start_class(c)+(i-1)*n_data_per_fold_of_class(c): ...
                   id_start_class(c)+(i-1)*n_data_per_fold_of_class(c)+remain_per_class(c)-1, :);
            cnt_last_fold = cnt_last_fold + remain_per_class(c);
        end
    end
     
    remain_per_class = remain_per_class - n_data_per_fold_of_class;
end

%% Make index for k-folds
for i = 1:k_folds-1
    start_id(i) = (i-1)*n_per_fold + 1;
    end_id(i) = i*n_per_fold;
end
start_id(k_folds) = (k_folds-1)*n_per_fold + 1;
end_id(k_folds) = size(x, 1);
end