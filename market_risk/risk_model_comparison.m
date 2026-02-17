%% RISK MODEL COMPARISON FRAMEWORK
% Portfolio: JPM, AAPL, MSFT, SPY
% Period: 2016–2026

clear; clc; close all;

%% ===============================
%% STEP 1 — Load Data
%% ===============================

data = readtable('equity_portfolio_data.csv');
data.Date = datetime(data.Date);

prices = [data.JPM data.AAPL data.MSFT data.SPY];
returns = diff(log(prices));

%% ===============================
%% STEP 2 — Portfolio Construction
%% ===============================

w = [0.25 0.25 0.25 0.25]';
portfolioReturns = returns * w;
losses = -portfolioReturns;

PortfolioValue = 1e6;

%% ===============================
%% STEP 3 — PARAMETRIC VaR
%% ===============================

mu = mean(portfolioReturns);
sigma = std(portfolioReturns);

z95 = norminv(0.05);
z99 = norminv(0.01);

VaR95_param = -(mu + z95*sigma);
VaR99_param = -(mu + z99*sigma);

%% ===============================
%% STEP 4 — HISTORICAL VaR & ES
%% ===============================

VaR95_hist = quantile(losses,0.95);
VaR99_hist = quantile(losses,0.99);

ES95_hist = mean(losses(losses > VaR95_hist));
ES99_hist = mean(losses(losses > VaR99_hist));

%% ===============================
%% STEP 5 — EVT (GPD) VaR & ES
%% ===============================

threshold = quantile(losses,0.95);
excessLosses = losses(losses > threshold) - threshold;

[paramEsts, ~] = gpfit(excessLosses);

xi = paramEsts(1);
beta = paramEsts(2);

N = length(losses);
Nu = length(excessLosses);

alpha = 0.99;

VaR99_evt = threshold + ...
    (beta/xi)*(((N/Nu)*(1-alpha))^(-xi) - 1);

ES99_evt = (VaR99_evt + beta - xi*threshold) / (1 - xi);

%% ===============================
%% STEP 6 — Convert to Dollar Terms
%% ===============================

results = table;

results.Model = ["Parametric 95%"; "Parametric 99%"; ...
                 "Historical 95%"; "Historical 99%"; ...
                 "EVT 99%"];

results.VaR_Return = [VaR95_param;
                      VaR99_param;
                      VaR95_hist;
                      VaR99_hist;
                      VaR99_evt];

results.ES_Return = [NaN;
                     NaN;
                     ES95_hist;
                     ES99_hist;
                     ES99_evt];

results.VaR_Dollar = results.VaR_Return * PortfolioValue;
results.ES_Dollar  = results.ES_Return  * PortfolioValue;

%% ===============================
%% DISPLAY RESULTS
%% ===============================

disp(' ');
disp('========= RISK MODEL COMPARISON =========');
disp(results);

fprintf('\nGPD Shape (xi): %.4f\n', xi);
fprintf('GPD Scale (beta): %.4f\n', beta);


