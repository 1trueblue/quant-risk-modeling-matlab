%% EXPECTED SHORTFALL PROJECT

clear; clc; close all;

%% STEP 1 — Load Data
data = readtable('equity_portfolio_data.csv');
data.Date = datetime(data.Date);

prices = [data.JPM data.AAPL data.MSFT data.SPY];

returns = diff(log(prices));

%% STEP 2 — Portfolio Setup
w = [0.25 0.25 0.25 0.25]';
portfolioReturns = returns * w;

%% STEP 3 — Define Confidence Level
alpha = 0.05;   % 95%

%% STEP 4 — Compute Historical VaR
VaR = quantile(portfolioReturns, alpha);

%% STEP 5 — Compute Expected Shortfall
tailLosses = portfolioReturns(portfolioReturns < VaR);

ES = mean(tailLosses);

%% STEP 6 — Convert to Dollar Terms
PortfolioValue = 1e6;

DollarVaR = -VaR * PortfolioValue;
DollarES  = -ES  * PortfolioValue;

%% STEP 7 — Display Results

fprintf('----- HISTORICAL RISK MEASURES -----\n');
fprintf('VaR (95%%): %.5f\n', VaR);
fprintf('ES  (95%%): %.5f\n\n', ES);

fprintf('Dollar VaR: $%.2f\n', DollarVaR);
fprintf('Dollar ES : $%.2f\n', DollarES);

%% STEP 8 — Plot

figure;
histogram(portfolioReturns,50);
hold on;

xline(VaR,'r','LineWidth',2);
xline(ES,'k','LineWidth',2);

legend('Returns','VaR','Expected Shortfall');
title('Historical Expected Shortfall');
grid on;
