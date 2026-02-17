%% MANUAL GARCH(1,1) VaR

clear; clc; close all;

%% STEP 1 — Load Data
data = readtable('equity_portfolio_data.csv');
prices = [data.JPM data.AAPL data.MSFT data.SPY];
returns = diff(log(prices));

%% STEP 2 — Portfolio Returns
w = [0.25 0.25 0.25 0.25]';
r = returns * w;

T = length(r);

%% STEP 3 — Negative Log Likelihood Function

garch_nll = @(params) garchLikelihood(params, r);

% Initial guess: [omega alpha beta]
initParams = [1e-6 0.05 0.9];

options = optimset('Display','off');
estParams = fmincon(garch_nll, initParams, ...
    [],[],[],[], ...
    [1e-8 0 0], [Inf 1 1], [], options);

omega = estParams(1);
alpha = estParams(2);
beta  = estParams(3);

%% STEP 4 — Compute Conditional Variance

sigma2 = zeros(T,1);
sigma2(1) = var(r);

for t = 2:T
    sigma2(t) = omega + alpha*r(t-1)^2 + beta*sigma2(t-1);
end

sigma = sqrt(sigma2);

%% STEP 5 — Compute GARCH VaR (95%)

z = norminv(0.05);

VaR_garch = z .* sigma;

%% STEP 6 — Backtest

violations = r < VaR_garch;
numViolations = sum(violations);
expectedViolations = 0.05*T;

fprintf('Observed Violations: %d\n', numViolations);
fprintf('Expected Violations: %.2f\n', expectedViolations);

%% STEP 7 — Kupiec Test

x = numViolations;
p = 0.05;

LR = -2 * ( ...
    log(((1-p)^(T-x)) * (p^x)) - ...
    log(((1 - x/T)^(T-x)) * ((x/T)^x)) ...
    );

pValue = 1 - chi2cdf(LR,1);

fprintf('Kupiec p-value: %.4f\n', pValue);

%% STEP 8 — Plot

figure;
plot(data.Date(2:end), r);
hold on;
plot(data.Date(2:end), VaR_garch, 'r','LineWidth',1.5);
title('Manual GARCH(1,1) VaR');
legend('Returns','GARCH VaR');
grid on;

%% ===============================
%% GARCH Likelihood Function
%% ===============================
function nll = garchLikelihood(params, r)

omega = params(1);
alpha = params(2);
beta  = params(3);

T = length(r);
sigma2 = zeros(T,1);
sigma2(1) = var(r);

for t = 2:T
    sigma2(t) = omega + alpha*r(t-1)^2 + beta*sigma2(t-1);
end

% Ensure positivity
if any(sigma2 <= 0)
    nll = Inf;
    return;
end

logL = -0.5 * sum( log(2*pi) + log(sigma2) + (r.^2)./sigma2 );
nll = -logL;

end
