%% HISTORICAL VAR PROJECT
% Estimate Historical Value at Risk (VaR)
% Dataset: equity_portfolio_data.csv
% Assets: JPM, AAPL, MSFT, SPY
% Period: 2016–2026

clear; clc; close all;

%% STEP 1 — Load Data
data = readtable('equity_portfolio_data.csv');

% Convert Date (if needed)
data.Date = datetime(data.Date);

dates = data.Date;

% Extract adjusted close prices
prices = [data.JPM data.AAPL data.MSFT data.SPY];

%% STEP 2 — Compute Log Returns
returns = diff(log(prices));      % 2639 × 4
dates_ret = dates(2:end);

%% STEP 3 — Define Portfolio Weights
% Equal weight portfolio
w = [0.25 0.25 0.25 0.25]';

%% STEP 4 — Compute Portfolio Returns
portfolioReturns = returns * w;   % 2639 × 1

%% STEP 5 — Historical VaR Calculation

alpha95 = 0.05;
alpha99 = 0.01;

% Using quantile
VaR95 = quantile(portfolioReturns, alpha95);
VaR99 = quantile(portfolioReturns, alpha99);

%% STEP 6 — Convert to Dollar VaR

PortfolioValue = 1e6;   % $1,000,000

DollarVaR95 = -VaR95 * PortfolioValue;
DollarVaR99 = -VaR99 * PortfolioValue;

%% STEP 7 — Display Results

fprintf('----- HISTORICAL VaR RESULTS -----\n');
fprintf('95%% Historical VaR (Return): %.5f\n', VaR95);
fprintf('99%% Historical VaR (Return): %.5f\n\n', VaR99);

fprintf('95%% Historical VaR (Dollar): $%.2f\n', DollarVaR95);
fprintf('99%% Historical VaR (Dollar): $%.2f\n', DollarVaR99);

%% STEP 8 — Visualize Distribution

figure;
histogram(portfolioReturns, 50);
hold on;

xline(VaR95, 'r', 'LineWidth', 2);
xline(VaR99, 'k', 'LineWidth', 2);

legend('Portfolio Returns', '95% VaR', '99% VaR');
title('Historical VaR — Portfolio Return Distribution');
xlabel('Daily Return');
ylabel('Frequency');
grid on;
