function [W, obj, time] = PMFSLRS(X, Y, opt)
%% function discription
% ----------------------------------------------------------------------
% Inputs:   
%          X: A d * n array, the i-th instance is stored in X(:,i)
%          Y: A c * n array, c is the number of possible labels, Y(j,i) is 1 if the i-th instance has the j-th label, and 0 otherwise
%
%          opt: the set of parameters
%
% Outputs:  
%          W: A d * c array, d is the number of features, coefficient matrix for feature selection
%
%          obj: The objective values
%          
%          time--PMFS-LRS's running time
% ----------------------------------------------------------------------

alpha = opt.alpha;  
beta = opt.beta;  
lambda = opt.lambda;
mu=1e8;
max_iter = opt.max_iter;
miniLossMargin = opt.minimumLossMargin;

[num_feature,num_sample]=size(X); 
num_label = size(Y, 1);
L = Laplacian(X');
obj = [];
H=eye(num_sample)-ones(num_sample,num_sample)./num_sample;

%% initialization

% Initialize W, F and N
W = randn(num_feature, num_label);
F = Y;
N = zeros(num_label,num_sample);

Lip1 = norm(2*H+2*L+mu*eye(num_sample),'fro');
Lip2 = 1;
 
iter = 1;
W21 = sum(sqrt(sum(W.*W,2)));
CorrelationLoss  = trace(F*L*F');
traceOfF    = trace(sqrt(F'*F));
sparesN    = sum(sum(abs(N)));
obj(iter)= norm((W'*X-F)*H,'fro')^2 + CorrelationLoss + alpha*traceOfF + beta*sparesN + lambda*W21;
oldloss = obj(iter);

tic;
while iter <= max_iter
%% update W  
d = 0.5./sqrt(sum(W.*W, 2) + eps);
D = diag(d);
W = (X * H * X' + lambda * D + eps*eye(num_feature)) \ (X * H * F'); 

%% update F
Gf = 2*F*H + 2*F*L + mu*F + mu*N - mu*Y- 2*W'*X*H;
F1 = F - 1/Lip1.*Gf;
[M,S,Fhat] = svd(F1,'econ');   
sp = diag(S);   
svp = length(find(sp>alpha/Lip1));    
if svp>=1
   sp = sp(1:svp)-alpha/Lip1;
else
   svp=1;
   sp=0;
end
Fhat =  M(:,1:svp)*diag(sp)*Fhat(:,1:svp)' ;    
F = Fhat; 

%% update N       
Gn = F + N -Y;
N1 = N - 1/Lip2.*Gn;
N   = softthres(N1,beta/(mu*Lip2));

%% Loss
W21 = sum(sqrt(sum(W.*W,2)+eps));
CorrelationLoss  = trace(F*L*F');
traceOfF    = trace(sqrt(F'*F));
sparesN    = sum(sum(abs(N)));

totalloss = (norm((W'*X - F)*H, 'fro'))^2 + CorrelationLoss + alpha*traceOfF + beta*sparesN + lambda*W21;
iter=iter+1;
obj(iter)=totalloss;

% convergence condition
if abs((oldloss - totalloss)/oldloss) <= miniLossMargin && iter >= 5
   break;
end
if totalloss <=0
  break;
end
oldloss = totalloss;
end   
time = toc;
end

function W = softthres(W_t,lambda)
    W = sign(W_t).*max(W_t-lambda, 0);  
end


