function [C, sigma] = dataset3Params(X, y, Xval, yval)
% You need to return the following variables correctly.
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_array = [0.01;0.03;0.1;0.3;1;3;10;30];     %C??
sigma_array = [0.01;0.03;0.1;0.3;1;3;10;30]; %sigma??
error_array = zeros(8,8);                    %??????????i?j??????C?i??sigma(j)??????
error_min = 10000;                           %?????????
 



for i = 1:8,
    for j = 1:8,
        model= svmTrain(X, y, C_array(i), @(x1, x2) gaussianKernel(x1, x2, sigma_array(j)));   %?C(i),sigma(j)?????SVM??X?y??????
        predictions = svmPredict(model, Xval);                                     %?????????model???????????
        error_array(i,j) =  mean(double(predictions ~= yval));                     %?????
        if(error_array(i,j) < error_min)                                           %??????????????????????C?sigma
            error_min = error_array(i,j);
            C = C_array(i);
            sigma = sigma_array(j);
        end
    end
end
% fprintf('The training C    The training sigma    error\n');                 %?????C,sigma??????????????????????
% for i = 1:8,
%     for j = 1:8,
%        fprintf('%f   %f   %f\n',C_array(i),sigma_array(j),error_array(i,j));               %??C(i)?sigma(j)??????
%     end
% end
% fprintf('%f and %f perform best, the error is %f',C,sigma,error_min);                      %???????C,sigma???????????
% % =========================================================================
end

