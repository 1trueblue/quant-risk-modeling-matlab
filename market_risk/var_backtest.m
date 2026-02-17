%% VAR BACKTESTING PROJECT (Corrected)

clear; clc; close all;

%% STEP 1 — Load Data
data = readtable('/Users/tejasbakre634/Desktop/MATLAB_Fin/equity_portfolio_data.csv');
data.Date = datetime(data.Date);

prices = [data.JPM data.AAPL data.MSFT data.SPY];
dates = data.Date;

returns = diff(log(prices));
dates_ret = dates(2:end);

%% STEP 2 — Portfolio Setup
w = [0.25 0.25 0.25 0.25]';
portfolioReturns = returns * w;

%% STEP 3 — Rolling Window Setup
window = 250;        % 1-year window
alpha = 0.05;        % 95% VaR
z = norminv(alpha);

numObs = length(portfolioReturns);

% Preallocate
VaR = zeros(numObs - window, 1);

%% STEP 4 — Rolling VaR Calculation
for t = window+1:numObs
    
    windowData = portfolioReturns(t-window:t-1);
    
    mu = mean(windowData);
    sigma = std(windowData);
    
    VaR(t-window) = mu + z * sigma;
    
end

%% STEP 5 — Align Actual Returns
actualReturns = portfolioReturns(window+1:end);

%% STEP 6 — Count Violations
violations = actualReturns < VaR;

numViolations = sum(violations);
expectedViolations = alpha * length(actualReturns);

fprintf('Observed Violations: %d\n', numViolations);
fprintf('Expected Violations: %.2f\n', expectedViolations);

%% STEP 7 — Kupiec Test
T = length(actualReturns);
x = numViolations;
p = alpha;

LR = -2 * ( ...
    log(((1-p)^(T-x)) * (p^x)) - ...
    log(((1 - x/T)^(T-x)) * ((x/T)^x)) ...
    );

pValue = 1 - chi2cdf(LR,1);

fprintf('Kupiec Test Statistic: %.4f\n', LR);
fprintf('Kupiec p-value: %.4f\n', pValue);


%% =====================================
%% CHRISTOFFERSEN INDEPENDENCE TEST
%% =====================================

% Violation series (already computed)
I = violations;

% Remove first observation (no previous value)
I_lag = I(1:end-1);
I_curr = I(2:end);

% Transition counts
n00 = sum((I_lag == 0) & (I_curr == 0));
n01 = sum((I_lag == 0) & (I_curr == 1));
n10 = sum((I_lag == 1) & (I_curr == 0));
n11 = sum((I_lag == 1) & (I_curr == 1));

% Transition probabilities
pi0 = n01 / (n00 + n01);
pi1 = n11 / (n10 + n11);
pi  = (n01 + n11) / (n00 + n01 + n10 + n11);

% Likelihood Ratio for Independence
LR_ind = -2 * ( ...
    log((1-pi)^(n00+n10) * pi^(n01+n11)) - ...
    log((1-pi0)^n00 * pi0^n01 * (1-pi1)^n10 * pi1^n11) ...
    );

pValue_ind = 1 - chi2cdf(LR_ind,1);

fprintf('\n--- CHRISTOFFERSEN INDEPENDENCE TEST ---\n');
fprintf('LR Statistic: %.4f\n', LR_ind);
fprintf('p-value: %.4f\n', pValue_ind);

%% =====================================
%% CONDITIONAL COVERAGE TEST
%% =====================================

LR_cc = LR + LR_ind;
pValue_cc = 1 - chi2cdf(LR_cc,2);

fprintf('\n--- CONDITIONAL COVERAGE TEST ---\n');
fprintf('LR Statistic: %.4f\n', LR_cc);
fprintf('p-value: %.4f\n', pValue_cc);



%% STEP 8 — Plot
figure;
plot(dates_ret(window+1:end), actualReturns);
hold on;
plot(dates_ret(window+1:end), VaR, 'r','LineWidth',1.5);

legend('Actual Returns','VaR');
title('Rolling 95% VaR Backtest');
grid on;
