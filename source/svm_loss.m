function [loss, dL] = svm_loss(pred, y)
batchsize = size(pred, 1);
N_CLASS = 10;
class_pred = zeros(batchsize, 1);
for i = 1:batchsize
    class_pred(i) = pred(i, y(i));
end
margins = max(0, pred - repmat(class_pred, 1, N_CLASS) + 1);

for i = 1:batchsize
    margins(i, y(i)) = 0;
end
loss = sum(sum(margins))/batchsize;
num_pos = sum(margins > 0, 2);
dL = zeros(size(pred));
dL(margins > 0) = 1;
for i = 1:batchsize
    dL(i, y(i)) = dL(i, y(i)) - num_pos(i);
end
dL = dL/batchsize;
end
