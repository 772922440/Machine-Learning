function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


% for i = 1 : m
%     h_theta = sigmoid(X(i,:)*theta); %right  h_theta = scalar 
     
%     temp = temp + ((-y(i) * log(h_theta)) - (1-y(i))*(log(1-h_theta))); 
%     

%     theta = (h_theta-y(i)) .* X(i,:)';  % 5 *1
%     grad = grad + theta;
%    
%     fprintf('H:%f T:%f TH:%f %f %f\n', h_theta, temp, theta(1),theta(2),theta(3));
  
% end


 hx = sigmoid(X * theta);  
 
 %fprintf('hx: %f\n',hx);
 
 J = (1.0/m) * sum(-y .* log(hx) - (1.0 - y) .* log(1.0 - hx));  
 
 grad = (1.0/m) .* X' * (hx - y);  


% =============================================================

end
