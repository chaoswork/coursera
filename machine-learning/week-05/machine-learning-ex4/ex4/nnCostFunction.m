function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X_new = [ones(m,1) X];

a_2 = sigmoid(X_new*Theta1');

h = sigmoid([ones(size(a_2,1),1) a_2]*Theta2');

for i = 1:num_labels
  J = J + (-1*(log(h(:,i)))'*(y==i) - (log(1-h(:,i)))'*(1-(y==i)) );
endfor

J = J/m;
th1 = Theta1(:,2:size(Theta1,2))(:);
th2 = Theta2(:,2:size(Theta2,2))(:);
J = J + lambda*(th1'*th1 + th2'*th2)/2/m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for t = 1:m
  a_1 = X_new(t,:)'; % 401 1
  z_2 = Theta1*a_1; % 25 1
  a_2 = [1; sigmoid(z_2)]; % 26 1
  z_3 = Theta2*a_2; % 10 1
  a_3 = sigmoid(z_3); % 10 1

  y_loga = 1:num_labels;
  y_loga = y_loga' == y(t,:); % 10 1
  delta_3 = a_3 - y_loga; % 10 1
  delta_2 = (Theta2'*delta_3)(2:end).*sigmoidGradient(z_2); % 25 1
  Theta2_grad = Theta2_grad + delta_3*(a_2');  %10 26
  Theta1_grad = Theta1_grad + delta_2*(a_1');  %25 401
endfor

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

reg_1 = lambda*Theta1/m;
reg_1(:,1) = 0;
Theta1_grad = Theta1_grad + reg_1;
reg_2 = lambda*Theta2/m;
reg_2(:,1) = 0;
Theta2_grad = Theta2_grad + reg_2;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
