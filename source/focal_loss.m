function [loss, dL] = focal_loss(pred, y, gamma, alpha)
% Utility function: Cross entropy
% Param:
%   pred: Output of last layer, shape of Cx1
%   y: Ground truth, shape of Cx1 (one-hot vector)
% Return:
%   loss: Loss value
%   dL: Gradient with loss

p = softmax(pred'); % Shape of CxN
batchsize = size(pred, 1);
log_likelihood = zeros(batchsize, 1);

for bs = 1:batchsize
    log_likelihood(bs) = -alpha * (1 - p(y(bs)))^gamma * log(p(y(bs)));
end

loss = log_likelihood;
dL = p';
for bs =1:batchsize
    for j = 1:10
        if j == y(bs)
            dL(bs,j) = alpha * (1 - dL(bs,j))^gamma * (gamma * dL(bs,j)*log(dL(bs,j)) + dL(bs,j) - 1);
        else
            dL(bs,j) = alpha * (1 - dL(bs,j))^(gamma-1)* dL(bs,j) * (1 - dL(bs,j) - gamma * dL(bs,j)*log(dL(bs,j)));
        end
    end
end
end