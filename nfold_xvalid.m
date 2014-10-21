function [ learning_stats ] = nfold_xvalid( all_inputs, N, method)
%% N-fold cross-validation

% Prealloc the return function
learning_stats = zeros(N,2);

% Total input vectors
num_inputs = numel(all_inputs(:,1));

% Calculate number of train and test for each step %note: this method currently loses 2*N samples
num_train = ceil(num_inputs * .9);  %% This is why we lose (rounding)
num_valid = num_inputs - num_train;

% Prealloc arrays
train_set = zeros (num_train,257);
valid_set = zeros (num_valid,257);

%% Shuffle the dataset
shuffled_inputs = all_inputs(randperm(numel(all_inputs(:,1))),:);

for validation = 1:N
    
    %validation indices
    valid_start = (validation - 1) * num_valid + 1;
    valid_end = validation * num_valid;
    
    %training indices
    train_set_lo_start = 1;
    train_set_lo_end = valid_start - 1;
    train_set_hi_start = valid_end + 1;
    train_set_hi_end = num_inputs;

    % Make training set
    if train_set_lo_end == 0 
        train_set = shuffled_inputs(train_set_hi_start:train_set_hi_end,:);
    elseif train_set_hi_start > num_inputs
        train_set = shuffled_inputs(train_set_lo_start:train_set_lo_end,:);
    else
        train_set_lo = shuffled_inputs(train_set_lo_start:train_set_lo_end,:);
        train_set_hi = shuffled_inputs(train_set_hi_start:train_set_hi_end,:);
        train_set = [train_set_lo;train_set_hi];
    end
    
    % Make validation set
    valid_set = shuffled_inputs(valid_start:valid_end,:);
    
    %% Use the machine
    if strcmp(method, 'SVM')
       % svm_train(train_set);
       % store lerning curve
       % svm_valid(valid_set);
       % store validation curve
       
    elseif strcmp(method, 'DNN')
       % svm_train(train_set);
       % store lerning curve in learning_stats
       % svm_valid(valid_set);
       % store validation curve in learning_stats
        
    elseif strcmp(method, 'XXX')
       % svm_train(train_set);
       % store lerning curve in learning_stats
       % svm_valid(valid_set);
       % store validation curve in learning_stats
            
    elseif strcmp(method, 'YYY')
       % svm_train(train_set);
       % store lerning curve in learning_stats
       % svm_valid(valid_set);
       % store validation curve in learning_stats
                
    elseif strcmp(method, 'ZZZ')
       % svm_train(train_set);
       % store lerning curve in learning_stats
       % svm_valid(valid_set);
       % store validation curve in learning_stats
        
    end    
end