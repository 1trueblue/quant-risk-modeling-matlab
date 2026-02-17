%% GARCH(1,1) VaR BACKTEST

clear; clc; close all;

%% STEP 1 — Load Data
data = readtable('equity_portfolio_data.csv');
data.Date = datetime(data.Date);

prices = [data.JPM data.AAPL data.MSFT data.SPY];
returns = diff(log(prices));

%% STEP 2 — Portfolio Returns
w = [0.25 0.25 0.25 0.25]';
portfolioReturns = returns * w;

%% STEP 3 — Fit GARCH(1,1)
model = garch(1,1);

% Estimate model parameters
[estModel, ~] = estimate(model, portfolioReturns, 'Display','off');

% Infer conditional variances
[condVar, ~] = infer(estModel, portfolioReturns);

condSigma = sqrt(condVar);

%% STEP 4 — Compute VaR (95%)
alpha = 0.05;
z = norminv(alpha);

VaR_garch = estModel.Constant + z .* condSigma;

%% STEP 5 — Align for Backtest
actualReturns = portfolioReturns;
violations = actualReturns < VaR_garch;

numViolations = sum(violations);
expectedViolations = alpha * length(actualReturns);

fprintf('GARCH VaR Backtest:\n');
fprintf('Observed Violations: %d\n', numViolations);
fprintf('Expected Violations: %.2f\n', expectedViolations);

%% STEP 6 — Kupiec Test

T = length(actualReturns);
x = numViolations;
p = alpha;

LR = -2 * ( ...
    log(((1-p)^(T-x)) * (p^x)) - ...
    log(((1 - x/T)^(T-x)) * ((x/T)^x)) ...
    );

pValue = 1 - chi2cdf(LR,1);

fprintf('Kupiec p-value: %.4f\n', pValue);


%% =====================================
%% CHRISTOFFERSEN INDEPENDENCE TEST
%% =====================================

I = violations;

I_lag = I(1:end-1);
I_curr = I(2:end);

n00 = sum((I_lag == 0) & (I_curr == 0));
n01 = sum((I_lag == 0) & (I_curr == 1));
n10 = sum((I_lag == 1) & (I_curr == 0));
n11 = sum((I_lag == 1) & (I_curr == 1));

pi0 = n01 / (n00 + n01);
pi1 = n11 / (n10 + n11);
pi  = (n01 + n11) / (n00 + n01 + n10 + n11);

LR_ind = -2 * ( ...
    log((1-pi)^(n00+n10) * pi^(n01+n11)) - ...
    log((1-pi0)^n00 * pi0^n01 * (1-pi1)^n10 * pi1^n11) ...
    );

pValue_ind = 1 - chi2cdf(LR_ind,1);

fprintf('\n--- GARCH CHRISTOFFERSEN INDEPENDENCE ---\n');
fprintf('LR Statistic: %.4f\n', LR_ind);
fprintf('p-value: %.4f\n', pValue_ind);

%% =====================================
%% CONDITIONAL COVERAGE TEST
%% =====================================

LR_cc = LR + LR_ind;
pValue_cc = 1 - chi2cdf(LR_cc,2);

fprintf('\n--- GARCH CONDITIONAL COVERAGE ---\n');
fprintf('LR Statistic: %.4f\n', LR_cc);
fprintf('p-value: %.4f\n', pValue_cc);



%% STEP 7 — Plot

figure;
plot(data.Date(2:end), actualReturns);
hold on;
plot(data.Date(2:end), VaR_garch, 'r','LineWidth',1.5);

legend('Actual Returns','GARCH VaR');
title('GARCH(1,1) 95% VaR Backtest');
grid on;
