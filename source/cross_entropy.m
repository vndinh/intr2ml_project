function [loss, dL] = cross_entropy(pred, y)
% Utility function: Cross entropy
% Param:
%   pred: Output of last layer, shape of NxC
%   y: Ground truth, shape of Nx1
% Return:
%   loss: Loss value
%   dL: Gradient with loss

p = softmax(pred'); % Shape of CxN
batchsize = size(pred, 1);
log_likelihood = zeros(batchsize, 1);

for i = 1:batchsize
    log_likelihood(i) = -log(p(y(i)));
end

loss = log_likelihood;
dL = p';
for i =1:batchsize
    dL(i,y(i)) = dL(i,y(i)) - 1;
end
end