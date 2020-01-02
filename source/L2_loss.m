function [loss, dL] = L2_loss(pred, y)
batchsize = size(pred, 1);
for i = 1:batchsize
    loss = 0.5 * sum((pred(y(i)) - 1).^2);
    dL = pred;
    dL(y) = dL(y) - 1;
end
end