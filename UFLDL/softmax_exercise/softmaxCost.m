function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

p_matrix = theta * data;

% prevent overflow
p_matrix = bsxfun(@minus, p_matrix, max(p_matrix, [], 1));
p = exp(p_matrix);  
%normalization_factor = repmat(sum(p_matrix),numClasses,1);
%p = log(p_matrix./normalization_factor);

% compute the hypothesis
p = bsxfun(@rdivide, p, sum(p));

J_cost = (-1/numCases)*sum(sum(groundTruth.*log(p)));
J_weight = (1/2)*sum(sum(theta.*theta));
cost = J_cost + lambda*J_weight;

thetagrad = (-1/numCases)*((groundTruth - p)*data') + lambda*theta;








% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

