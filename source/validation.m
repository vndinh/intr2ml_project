%% validation code here
function res = validation(model, mode, data_input, ground_truth)

N_valid = size(data_input, 1);
valid_p = zeros(N_valid, 1);

if nargin == 3
    for idx = 1 : N_valid
        data_point = data_input(idx, :);
        [~, estimate] = feed_forward(model, mode, data_point);
        valid_p(idx) = estimate;
    end
    res = valid_p;
else
    for idx = 1 : N_valid
        data_point = data_input(idx, :);
        [~, estimate] = feed_forward(model, mode, data_point);
        valid_p(idx) = estimate;
    end
    res = mean(ground_truth == valid_p)*100;
end

end