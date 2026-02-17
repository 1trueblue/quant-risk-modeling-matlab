%% EXTREME VALUE THEORY (GPD) VaR & ES

clear; clc; close all;

%% STEP 1 — Load Data
data = readtable('equity_portfolio_data.csv');
data.Date = datetime(data.Date);

prices = [data.JPM data.AAPL data.MSFT data.SPY];
returns = diff(log(prices));

%% STEP 2 — Portfolio Returns
w = [0.25 0.25 0.25 0.25]';
portfolioReturns = returns * w;

%% STEP 3 — Convert Returns to Losses
losses = -portfolioReturns;

%% STEP 4 — Choose Threshold (95th percentile of losses)
threshold = quantile(losses, 0.95);

excessLosses = losses(losses > threshold) - threshold;

%% STEP 5 — Fit GPD to Tail
[paramEsts, paramCI] = gpfit(excessLosses);

xi = paramEsts(1);     % Shape parameter
beta = paramEsts(2);   % Scale parameter

%% STEP 6 — EVT VaR Calculation (99%)
alpha = 0.99;

N = length(losses);
Nu = length(excessLosses);

VaR_evt = threshold + ...
    (beta/xi)*(((N/Nu)*(1-alpha))^(-xi) - 1);

%% STEP 7 — EVT Expected Shortfall
ES_evt = (VaR_evt + beta - xi*threshold) / (1 - xi);

%% STEP 8 — Convert to Dollar
PortfolioValue = 1e6;

DollarVaR_evt = VaR_evt * PortfolioValue;
DollarES_evt  = ES_evt  * PortfolioValue;

%% STEP 9 — Display Results

fprintf('----- EVT RESULTS (99%%) -----\n');
fprintf('Shape (xi): %.4f\n', xi);
fprintf('Scale (beta): %.4f\n\n', beta);

fprintf('EVT VaR (99%%): %.5f\n', VaR_evt);
fprintf('EVT ES  (99%%): %.5f\n\n', ES_evt);

fprintf('Dollar VaR: $%.2f\n', DollarVaR_evt);
fprintf('Dollar ES : $%.2f\n', DollarES_evt);
