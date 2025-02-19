% Calculates the bias and variance of proposed permutation-invariant method
% and the normal kernel ridge regression as n increases in set incriments.
% e.g. 1000, 2000, and so on.

% Each section is a different situation, with either fixed bandwidth or the
% bandwidth being chosen as the optimal asymptotic bandwidth.

% In every example, the data are distributed IID N(0,1)

% All of the examples currently use a triangular kernel but I have other
% kernel options saved at the bottom of the file if you want to use
% something like the Gaussian kernel instead.

%% Simulation -1: d = 4, MSE optimal bandwidth

clear
clc
rng(8)

% Evaluation point. We are trying to estimate f(point)
%point = [0, 0.25, 0.5, 0.75];
%permutations = [point;circshift(point,1);circshift(point,2);circshift(point,3)];

% point = [0, 0.25, 0.5, 0.75, 0.75];
% permutations = [0, 0.25, 0.5, 0.75, 0.75;
%     0.75, 0, 0.25, 0.5, 0.75;
%     0.5, 0.75, 0, 0.25, 0.75;
%     0.25, 0.5, 0.75, 0, 0.75];

%point = [0, 0.25, 0.5, 0.75, 1];
%permutations = [point;circshift(point,1);circshift(point,2);circshift(point,3); circshift(point,4)];

%point = [0, 0.25, 0.5];
%permutations = [point;circshift(point,1);circshift(point,2)];

point = [0, 0.5, 1];
permutations = [point;circshift(point,1);circshift(point,2)];

% Data parameters
d = length(point);
mu = zeros(1,d);
Sigma = eye(d); 

perm_estimate = zeros(1,d);

% Save the number of permutations. This is just d!
len = size(permutations,1);

% Set the number of replications
reps = 1000;

% Step size for n. Goes from n_step, 2*n_step, ...
n_step = 1000;
% Maximum number of n_step increases
n_num = 10;
% Sequence of n values
n_seq = linspace(n_step, n_step*n_num, n_num);

% Average bias and variance for each choice of n
bias = zeros(n_num, 1);
bias_perm = zeros(n_num, 1);
var = zeros(n_num, 1);
var_perm = zeros(n_num, 1);

for i = 1:n_num
    % n amount for this step. Increases each step by n_step
    n = n_seq(i);
    % MSE optimal bandwidth
    h = n^(-1/(d+4));

    % The estimated value of the KDE estimator for each replication
    f_hat_p = zeros(reps, 1);
    % The estimated value of the KDE permutation-invariant estimator for each replication
    f_hat_perm = zeros(reps, 1);

    for r = 1:reps
        % Generate random values of X ~ N(mu, Sigma)
        x = mvnrnd(mu, Sigma, n);
        
        % KDE estimate for f(point)
        f_hat_p(r) = 1/(n*h^d) * sum(triangle_kernel((x-point)./h));
        
        % Permutation-invariant KDE estimate for f(point)
        % First sums all values of the KDE for each permutation then
        % divides by the number of permutations        
        for j = 1:len
            perm_estimate(j) = 1/(n*h^d) * sum(triangle_kernel((x-permutations(j,:))./h));
        end
        f_hat_perm(r) = sum(perm_estimate) / len;
    end

    % Variance for each method
    var(i) = std(f_hat_p)^2;
    var_perm(i) = std(f_hat_perm)^2;

    % MSE for each method
    MSE(i) = mean((f_hat_p - mvnpdf(point, mu, Sigma)).^2);
    MSE_perm(i) = mean((f_hat_perm - mvnpdf(point, mu, Sigma)).^2);
    
    % Bias for each method
    bias(i) = mean(f_hat_p) - mvnpdf(point, mu, Sigma);
    bias_perm(i) = mean(f_hat_perm) - mvnpdf(point, mu, Sigma);

end

fprintf('%e & %e & %e & %e\n',bias(10), bias_perm(10), var(10), var_perm(10));

%% Simulation 0: d = 4, MSE optimal bandwidth

clear
clc
rng(8)


% Data parameters
d = 4;
mu = zeros(1,d);
Sigma = eye(d); 


% Evaluation point. We are trying to estimate f(point)
point = [0, 0.25, 0.5, 0.75];

% Generates all possible permutation of the evaluation point.
permutations = [point;circshift(point,1);circshift(point,2);circshift(point,3)];
% Save the number of permutations. This is just d!
len = length(permutations);

% Set the number of replications
reps = 1000;

% Step size for n. Goes from n_step, 2*n_step, ...
n_step = 1000;
% Maximum number of n_step increases
n_num = 10;
% Sequence of n values
n_seq = linspace(n_step, n_step*n_num, n_num);

% Average bias and variance for each choice of n
bias = zeros(n_num, 1);
bias_perm = zeros(n_num, 1);
var = zeros(n_num, 1);
var_perm = zeros(n_num, 1);

for i = 1:n_num
    % n amount for this step. Increases each step by n_step
    n = n_seq(i);
    % MSE optimal bandwidth
    h = n^(-1/(d+4));

    % The estimated value of the KDE estimator for each replication
    f_hat_p = zeros(reps, 1);
    % The estimated value of the KDE permutation-invariant estimator for each replication
    f_hat_perm = zeros(reps, 1);

    for r = 1:reps
        % Generate random values of X ~ N(mu, Sigma)
        x = mvnrnd(mu, Sigma, n);
        
        % KDE estimate for f(point)
        f_hat_p(r) = 1/(n*h^d) * sum(triangle_kernel((x-point)./h));
        
        % Permutation-invariant KDE estimate for f(point)
        % First sums all values of the KDE for each permutation then
        % divides by the number of permutations        
        for j = 1:len
            perm_estimate(j) = 1/(n*h^d) * sum(triangle_kernel((x-permutations(j,:))./h));
        end
        f_hat_perm(r) = sum(perm_estimate) / len;
    end

    % Variance for each method
    var(i) = std(f_hat_p)^2;
    var_perm(i) = std(f_hat_perm)^2;

    % MSE for each method
    MSE(i) = mean((f_hat_p - mvnpdf(point, mu, Sigma)).^2);
    MSE_perm(i) = mean((f_hat_perm - mvnpdf(point, mu, Sigma)).^2);
    
    % Bias for each method
    bias(i) = mean(f_hat_p) - mvnpdf(point, mu, Sigma);
    bias_perm(i) = mean(f_hat_perm) - mvnpdf(point, mu, Sigma);

end

% Plots variance of permutation-invariant method vs normal KDE method for
% each value of n and exports to a png
plot(n_seq, var_perm,'Linewidth',2)
hold on
plot(n_seq, var,'Linewidth',2,'LineStyle','--')
legend('Averaged KDE', 'Standard KDE')
title('Variance vs. n, for d = 4')
xlabel('n')
ylabel('Variance')
saveas(gcf, 'd=4_optimal_bandwidth_variance.png')

hold off

% Plots MSE of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, MSE_perm)
hold on
plot(n_seq, MSE)
legend('Permutation-Invariant', 'Permutation Variant')
title('MSE vs. N and h = o(N^{-1/8}) with d = 4')
xlabel('n')
ylabel('Mean Squared Error')
saveas(gcf, 'd=4_optimal_bandwidth_MSE.png')

hold off

% Plots bias of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, bias_perm,'Linewidth',2)
hold on
plot(n_seq, bias,'Linewidth',2,'LineStyle','--')
legend('Averaged KDE', 'Standard KDE')
title('Bias vs. n, for d = 4')
xlabel('n')
ylabel('Bias')
saveas(gcf, 'd=4_optimal_bandwidth_bias.png')

hold off


%% Simulation 1: d = 4, fixed bandwidth

clear
clc
rng(5)


% Data parameters
d = 4;
mu = zeros(1,d);
Sigma = eye(d); 


% Evaluation point. We are trying to estimate f(point)
point = [0, 0.25, 0.5, 0.75];

% Generates all possible permutation of the evaluation point.
permutations = perms(point);
% Save the number of permutations. This is just d!
len = length(permutations);

% Set the number of replications
reps = 1000;

% Step size for n. Goes from n_step, 2*n_step, ...
n_step = 1000;
% Maximum number of n_step increases
n_num = 10;
% Sequence of n values
n_seq = linspace(n_step, n_step*n_num, n_num);

% Average bias and variance for each choice of n
bias = zeros(n_num, 1);
bias_perm = zeros(n_num, 1);
var = zeros(n_num, 1);
var_perm = zeros(n_num, 1);

for i = 1:n_num
    % n amount for this step. Increases each step by n_step
    n = n_seq(i);
    % Fixed bandwidth
    h = 0.05;

    % The estimated value of the KDE estimator for each replication
    f_hat_p = zeros(reps, 1);
    % The estimated value of the KDE permutation-invariant estimator for each replication
    f_hat_perm = zeros(reps, 1);

    for r = 1:reps
        % Generate random values of X ~ N(mu, Sigma)
        x = mvnrnd(mu, Sigma, n);
        
        % KDE estimate for f(point)
        f_hat_p(r) = 1/(n*h^d) * sum(triangle_kernel((x-point)./h));
        
        % Permutation-invariant KDE estimate for f(point)
        % First sums all values of the KDE for each permutation then
        % divides by the number of permutations        
        for j = 1:len
            perm_estimate(j) = 1/(n*h^d) * sum(triangle_kernel((x-permutations(j,:))./h));
        end
        f_hat_perm(r) = sum(perm_estimate) / len;
    end

    % Variance for each method
    var(i) = std(f_hat_p)^2;
    var_perm(i) = std(f_hat_perm)^2;

    % MSE for each method
    MSE(i) = mean((f_hat_p - mvnpdf(point, mu, Sigma)).^2);
    MSE_perm(i) = mean((f_hat_perm - mvnpdf(point, mu, Sigma)).^2);
    
    % Bias for each method
    bias(i) = mean(f_hat_p) - mvnpdf(point, mu, Sigma);
    bias_perm(i) = mean(f_hat_perm) - mvnpdf(point, mu, Sigma);

end

% Plots variance of permutation-invariant method vs normal KDE method for
% each value of n and exports to a png
plot(n_seq, var_perm)
hold on
plot(n_seq, var)
legend('Permutation-Invariant', 'Permutation Variant')
title('Variance vs. N and fixed bandwidth with d = 4')
xlabel('N')
ylabel('Variance')
saveas(gcf, 'd=4_fixed_bandwidth_variance.png')

hold off

% Plots MSE of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, MSE_perm)
hold on
plot(n_seq, MSE)
legend('Permutation-Invariant', 'Permutation Variant')
title('MSE vs. N and fixed bandwidth with d = 4')
xlabel('N')
ylabel('Mean Squared Error')
saveas(gcf, 'd=4_fixed_bandwidth_MSE.png')

hold off

% Plots bias of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, bias_perm)
hold on
plot(n_seq, bias)
legend('Permutation-Invariant', 'Permutation Variant')
title('Bias vs. N and fixed bandwidth with d = 4')
xlabel('N')
ylabel('Bias')
saveas(gcf, 'd=4_fixed_bandwidth_bias.png')

hold off

%% Simulation 2: d = 4, MSE optimal bandwidth

clear
clc
rng(8)


% Data parameters
d = 4;
mu = zeros(1,d);
Sigma = eye(d); 


% Evaluation point. We are trying to estimate f(point)
point = [0, 0.25, 0.5, 0.75];

% Generates all possible permutation of the evaluation point.
permutations = perms(point);
% Save the number of permutations. This is just d!
len = length(permutations);

% Set the number of replications
reps = 1000;

% Step size for n. Goes from n_step, 2*n_step, ...
n_step = 1000;
% Maximum number of n_step increases
n_num = 10;
% Sequence of n values
n_seq = linspace(n_step, n_step*n_num, n_num);

% Average bias and variance for each choice of n
bias = zeros(n_num, 1);
bias_perm = zeros(n_num, 1);
var = zeros(n_num, 1);
var_perm = zeros(n_num, 1);

for i = 1:n_num
    % n amount for this step. Increases each step by n_step
    n = n_seq(i);
    % MSE optimal bandwidth
    h = n^(-1/(d+4));

    % The estimated value of the KDE estimator for each replication
    f_hat_p = zeros(reps, 1);
    % The estimated value of the KDE permutation-invariant estimator for each replication
    f_hat_perm = zeros(reps, 1);

    for r = 1:reps
        % Generate random values of X ~ N(mu, Sigma)
        x = mvnrnd(mu, Sigma, n);
        
        % KDE estimate for f(point)
        f_hat_p(r) = 1/(n*h^d) * sum(triangle_kernel((x-point)./h));
        
        % Permutation-invariant KDE estimate for f(point)
        % First sums all values of the KDE for each permutation then
        % divides by the number of permutations        
        for j = 1:len
            perm_estimate(j) = 1/(n*h^d) * sum(triangle_kernel((x-permutations(j,:))./h));
        end
        f_hat_perm(r) = sum(perm_estimate) / len;
    end

    % Variance for each method
    var(i) = std(f_hat_p)^2;
    var_perm(i) = std(f_hat_perm)^2;

    % MSE for each method
    MSE(i) = mean((f_hat_p - mvnpdf(point, mu, Sigma)).^2);
    MSE_perm(i) = mean((f_hat_perm - mvnpdf(point, mu, Sigma)).^2);
    
    % Bias for each method
    bias(i) = mean(f_hat_p) - mvnpdf(point, mu, Sigma);
    bias_perm(i) = mean(f_hat_perm) - mvnpdf(point, mu, Sigma);

end

% Plots variance of permutation-invariant method vs normal KDE method for
% each value of n and exports to a png
plot(n_seq, var_perm)
hold on
plot(n_seq, var)
legend('Permutation-Invariant', 'Permutation Variant')
title('Variance vs. N and h = o(N^{-1/8}) with d = 4')
xlabel('N')
ylabel('Variance')
saveas(gcf, 'd=4_optimal_bandwidth_variance.png')

hold off

% Plots MSE of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, MSE_perm)
hold on
plot(n_seq, MSE)
legend('Permutation-Invariant', 'Permutation Variant')
title('MSE vs. N and h = o(N^{-1/8}) with d = 4')
xlabel('N')
ylabel('Mean Squared Error')
saveas(gcf, 'd=4_optimal_bandwidth_MSE.png')

hold off

% Plots bias of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, bias_perm)
hold on
plot(n_seq, bias)
legend('Permutation-Invariant', 'Permutation Variant')
title('Bias vs. N and h = o(N^{-1/8}) with d = 4')
xlabel('N')
ylabel('Bias')
saveas(gcf, 'd=4_optimal_bandwidth_bias.png')

hold off


%% Simulation 3: d = 5, fixed bandwidth

clear
clc
rng(8)


% Data parameters
d = 5;
mu = zeros(1,d);
Sigma = eye(d); 


% Evaluation point. We are trying to estimate f(point)
point = [0, 0.25, 0.5, 0.75, 1];

% Generates all possible permutation of the evaluation point.
permutations = perms(point);
% Save the number of permutations. This is just d!
len = length(permutations);

% Set the number of replications
reps = 1000;

% Step size for n. Goes from n_step, 2*n_step, ...
n_step = 1000;
% Maximum number of n_step increases
n_num = 10;
% Sequence of n values
n_seq = linspace(n_step, n_step*n_num, n_num);

% Average bias and variance for each choice of n
bias = zeros(n_num, 1);
bias_perm = zeros(n_num, 1);
var = zeros(n_num, 1);
var_perm = zeros(n_num, 1);

for i = 1:n_num
    % n amount for this step. Increases each step by n_step
    n = n_seq(i);
    % Fixed bandwidth
    h = 0.1;

    % The estimated value of the KDE estimator for each replication
    f_hat_p = zeros(reps, 1);
    % The estimated value of the KDE permutation-invariant estimator for each replication
    f_hat_perm = zeros(reps, 1);

    for r = 1:reps
        % Generate random values of X ~ N(mu, Sigma)
        x = mvnrnd(mu, Sigma, n);
        
        % KDE estimate for f(point)
        f_hat_p(r) = 1/(n*h^d) * sum(triangle_kernel((x-point)./h));
        
        % Permutation-invariant KDE estimate for f(point)
        % First sums all values of the KDE for each permutation then
        % divides by the number of permutations        
        for j = 1:len
            perm_estimate(j) = 1/(n*h^d) * sum(triangle_kernel((x-permutations(j,:))./h));
        end
        f_hat_perm(r) = sum(perm_estimate) / len;
    end

    % Variance for each method
    var(i) = std(f_hat_p)^2;
    var_perm(i) = std(f_hat_perm)^2;

    % MSE for each method
    MSE(i) = mean((f_hat_p - mvnpdf(point, mu, Sigma)).^2);
    MSE_perm(i) = mean((f_hat_perm - mvnpdf(point, mu, Sigma)).^2);
    
    % Bias for each method
    bias(i) = mean(f_hat_p) - mvnpdf(point, mu, Sigma);
    bias_perm(i) = mean(f_hat_perm) - mvnpdf(point, mu, Sigma);

end

% Plots variance of permutation-invariant method vs normal KDE method for
% each value of n and exports to a png
plot(n_seq, var_perm)
hold on
plot(n_seq, var)
legend('Permutation-Invariant', 'Permutation Variant')
title('Variance vs. N and fixed bandwidth with d = 5')
xlabel('N')
ylabel('Variance')
saveas(gcf, 'd=5_fixed_bandwidth_variance.png')

hold off

% Plots MSE of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, MSE_perm)
hold on
plot(n_seq, MSE)
legend('Permutation-Invariant', 'Permutation Variant')
title('MSE vs. N and fixed bandwidth with d = 5')
xlabel('N')
ylabel('Mean Squared Error')
saveas(gcf, 'd=5_fixed_bandwidth_MSE.png')

hold off

% Plots bias of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, bias_perm)
hold on
plot(n_seq, bias)
legend('Permutation-Invariant', 'Permutation Variant')
title('Bias vs. N and fixed bandwidth with d = 5')
xlabel('N')
ylabel('Bias')
saveas(gcf, 'd=5_fixed_bandwidth_bias.png')

hold off

%% Simulation 4: d = 5, MSE optimal bandwidth

clear
clc
rng(8)


% Data parameters
d = 5;
mu = zeros(1,d);
Sigma = eye(d); 


% Evaluation point. We are trying to estimate f(point)
point = [0, 0.25, 0.5, 0.75, 1];

% Generates all possible permutation of the evaluation point.
permutations = perms(point);
% Save the number of permutations. This is just d!
len = length(permutations);

% Set the number of replications
reps = 1000;

% Step size for n. Goes from n_step, 2*n_step, ...
n_step = 1000;
% Maximum number of n_step increases
n_num = 10;
% Sequence of n values
n_seq = linspace(n_step, n_step*n_num, n_num);

% Average bias and variance for each choice of n
bias = zeros(n_num, 1);
bias_perm = zeros(n_num, 1);
var = zeros(n_num, 1);
var_perm = zeros(n_num, 1);

for i = 1:n_num
    % n amount for this step. Increases each step by n_step
    n = n_seq(i);
    % MSE optimal bandwidth
    h = 3 * n^(-1/(d+4));

    % The estimated value of the KDE estimator for each replication
    f_hat_p = zeros(reps, 1);
    % The estimated value of the KDE permutation-invariant estimator for each replication
    f_hat_perm = zeros(reps, 1);

    for r = 1:reps
        % Generate random values of X ~ N(mu, Sigma)
        x = mvnrnd(mu, Sigma, n);
        
        % KDE estimate for f(point)
        f_hat_p(r) = 1/(n*h^d) * sum(triangle_kernel((x-point)./h));
        
        % Permutation-invariant KDE estimate for f(point)
        % First sums all values of the KDE for each permutation then
        % divides by the number of permutations        
        for j = 1:len
            perm_estimate(j) = 1/(n*h^d) * sum(triangle_kernel((x-permutations(j,:))./h));
        end
        f_hat_perm(r) = sum(perm_estimate) / len;
    end

    % Variance for each method
    var(i) = std(f_hat_p)^2;
    var_perm(i) = std(f_hat_perm)^2;

    % MSE for each method
    MSE(i) = mean((f_hat_p - mvnpdf(point, mu, Sigma)).^2);
    MSE_perm(i) = mean((f_hat_perm - mvnpdf(point, mu, Sigma)).^2);
    
    % Bias for each method
    bias(i) = mean(f_hat_p) - mvnpdf(point, mu, Sigma);
    bias_perm(i) = mean(f_hat_perm) - mvnpdf(point, mu, Sigma);

end

% Plots variance of permutation-invariant method vs normal KDE method for
% each value of n and exports to a png
plot(n_seq, var_perm)
hold on
plot(n_seq, var)
legend('Permutation-Invariant', 'Permutation Variant')
title('Variance vs. N and h = o(N^{-1/8}) with d = 4')
xlabel('N')
ylabel('Variance')
saveas(gcf, 'd=5_optimal_bandwidth_variance.png')

hold off

% Plots MSE of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, MSE_perm)
hold on
plot(n_seq, MSE)
legend('Permutation-Invariant', 'Permutation Variant')
title('MSE vs. N and h = o(N^{-1/8}) with d = 4')
xlabel('N')
ylabel('Mean Squared Error')
saveas(gcf, 'd=5_optimal_bandwidth_MSE.png')

hold off

% Plots bias of permutation-invariant method vs normal KDE method for each
% value of n and exports to a png
plot(n_seq, bias_perm)
hold on
plot(n_seq, bias)
legend('Permutation-Invariant', 'Permutation Variant')
title('Bias vs. N and h = o(N^{-1/8}) with d = 4')
xlabel('N')
ylabel('Bias')
saveas(gcf, 'd=5_optimal_bandwidth_bias.png')

hold off

%% Real data (Australian Athletes Dataset):

clear
clc
rng(8)

data = readmatrix("ais.csv");
data = data(:,1:3);
for c = 1:3
    data(:,c) = (data(:,c) - min(data(:,c)))/(max(data(:,c)) - min(data(:,c)));
end
n_data_points = length(data);

% Evaluation point. We are trying to estimate f(point)
%point = [0, 0.25, 0.5, 0.75];
%permutations = [point;circshift(point,1);circshift(point,2);circshift(point,3)];

% point = [0, 0.25, 0.5, 0.75, 0.75];
% permutations = [0, 0.25, 0.5, 0.75, 0.75;
%     0.75, 0, 0.25, 0.5, 0.75;
%     0.5, 0.75, 0, 0.25, 0.75;
%     0.25, 0.5, 0.75, 0, 0.75];

%point = [0, 0.25, 0.5, 0.75, 1];
%permutations = [point;circshift(point,1);circshift(point,2);circshift(point,3); circshift(point,4)];

point = [0, 0.25, 0.5];
permutations = [point;circshift(point,1);circshift(point,2)];

%point = [0.25, 0.5, 1];
%permutations = [point;circshift(point,1);circshift(point,2)];

% Data parameters
d = length(point);
mu = zeros(1,d);
Sigma = eye(d); 

perm_estimate = zeros(1,d);

% Save the number of permutations. This is just d!
len = size(permutations,1);

% Set the number of replications
reps = 1000;

% Step size for n. Goes from n_step, 2*n_step, ...
n_step = 1000;
% Maximum number of n_step increases
%n_num = 10;
n_num = 1;

% Sequence of n values
%n_seq = linspace(n_step, n_step*n_num, n_num);
n_seq = [n_data_points];

% Average bias and variance for each choice of n
bias = zeros(n_num, 1);
bias_perm = zeros(n_num, 1);
var = zeros(n_num, 1);
var_perm = zeros(n_num, 1);

for i = 1:n_num
    % n amount for this step. Increases each step by n_step
    n = n_seq(i);
    % MSE optimal bandwidth
    h = n^(-1/(d+4));

    % The estimated value of the KDE estimator for each replication
    f_hat_p = zeros(reps, 1);
    % The estimated value of the KDE permutation-invariant estimator for each replication
    f_hat_perm = zeros(reps, 1);

    for r = 1:reps
        % Generate random values of X ~ N(mu, Sigma)
        sampled_rows = randi(n_data_points, n_data_points);
        x = data(sampled_rows, :);
        
        % KDE estimate for f(point)
        f_hat_p(r) = 1/(n*h^d) * sum(triangle_kernel((x-point)./h));
        
        % Permutation-invariant KDE estimate for f(point)
        % First sums all values of the KDE for each permutation then
        % divides by the number of permutations        
        for j = 1:len
            perm_estimate(j) = 1/(n*h^d) * sum(triangle_kernel((x-permutations(j,:))./h));
            %fprintf('%e', perm_estimate(j))
        end
        f_hat_perm(r) = sum(perm_estimate) / len;
    end

    % Variance for each method
    var(i) = std(f_hat_p)^2;
    var_perm(i) = std(f_hat_perm)^2;

    % MSE for each method
    MSE(i) = mean((f_hat_p - mvnpdf(point, mu, Sigma)).^2);
    MSE_perm(i) = mean((f_hat_perm - mvnpdf(point, mu, Sigma)).^2);
    
    % Bias for each method
    bias(i) = mean(f_hat_p) - mvnpdf(point, mu, Sigma);
    bias_perm(i) = mean(f_hat_perm) - mvnpdf(point, mu, Sigma);

end

fprintf('%e & %e & %e & %e\n',bias(n_num), bias_perm(n_num), var(n_num), var_perm(n_num));


%% Triangular kernel function

function f = triangle_kernel(x)
    v = max(0, 1 - abs(x));
    f = prod(v,2);
end

%% Parabolic kernel function

function f = parabolic_kernel(x)
    v = max(0, 3/4 .* (1 - x.^2));
    f = prod(v,2);
end

%% Gaussian kernel function

function f = gaussian_kernel(x)
    v = normpdf(x);
    f = prod(v,2);
end