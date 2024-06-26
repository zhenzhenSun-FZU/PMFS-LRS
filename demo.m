% This is an example file on how the PMFS-LRS [1] program could be used.

% [1] Sun Z, Chen Z, Liu J, et al.
% Partial multi-label feature selection via low-rank and sparse factorization with manifold learning[J]. Knowledge-Based Systems, 2024: 111899.

clc;clear;
addpath(genpath('.\'))
dataset = 'emotion';
path=strcat('./data/',dataset,'/');
datapath = strcat(path,'data.mat');
load(datapath);
targetpath = strcat(path,'target.mat');
load(targetpath);

% Noise rate
p = 3; %[3,5,7]
r = 1; %[1,2,3]

partialpath = strcat(path,'p',num2str(p),'r',num2str(r),'_noise_target.mat');
par = load(partialpath);
cs = struct2cell(par);
partial_labels = cell2mat(cs);

% Preprocess
data = zscore(data)';
target = target';
partial_labels = partial_labels';

% Calucate some statitics about the data
num_label = size(partial_labels,1);
[num_feature, num_data] = size(data);

feature_select = 5:5:50;
cv = 5;

% n_fold validation and evaluation
for i = 1:cv
    fprintf('Data processing, Cross validation: %d\n', i);
    
    % Training set and test set for the i-th fold
    n_test = round(num_data / cv);
    start_idx = (i-1)*n_test + 1;
    if i == cv
        test_idx = start_idx : num_data;
    else
        test_idx = start_idx:start_idx + n_test - 1;
    end
    II = 1:num_data;
    train_idx = setdiff(II, test_idx);
    train_data = data(:, train_idx);
    train_p_target = partial_labels(:, train_idx);
    train_target = target(:, train_idx);
    test_data = data(:, test_idx);
    test_target = target(:, test_idx);
    test_p_target = partial_labels(:, test_idx);
    
    opt.alpha = 10^4; opt.beta = 10^6; opt.lambda = 10^2; opt.max_iter = 50; opt.minimumLossMargin = 1e-5;
    
    % Running the PMFS-LRS procedure for feature selection
    t0 = clock;
    [W, obj, ~] = PMFSLRS(train_data, train_p_target, opt);
    time = etime(clock, t0);
    [dumb, idx] = sort(sum(W.*W,2),'descend');
    
    HL = []; RL= [];OE = [];CV = [];AP = [];
    for j = 1:length(feature_select)
        fprintf('Running the program with the selected features - %d/%d \n',feature_select(j),feature_select(length(feature_select)));
        f=idx(1:feature_select(j));
        test_prelab = W(f,:)'*test_data(f,:);
        RL(j) = Ranking_loss(test_prelab,test_target);
        OE(j) = One_error(test_prelab,test_target);
        CV(j) = coverage(test_prelab,test_target);
        AP(j) = Average_precision(test_prelab,test_target);
        bin_Pre_LD = binaryzation(softmax(test_prelab)',0.1);
        bin_test_target = binaryzation(softmax(test_target)',0.1);
        HL(j) = Hamming_loss(bin_Pre_LD',bin_test_target');
    end
end


