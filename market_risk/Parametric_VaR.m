ver
clear; clc; close all;

data = readtable('/Users/tejasbakre634/Desktop/MATLAB_Fin/equity_portfolio_data.csv');

head(data)
data.Date = datetime(data.Date);
dates = data.Date;

prices = [data.JPM data.AAPL data.MSFT data.SPY];
returns = diff(log(prices));
meanReturns = mean(returns);
volatility = std(returns);
corrMatrix = corr(returns);
covMatrix = cov(returns);

meanReturns
volatility
corrMatrix


w = [0.25 0.25 0.25 0.25]';
mu_p = w' * meanReturns';
sigma_p = sqrt(w' * covMatrix * w);

alpha95 = 0.05;
alpha99 = 0.01;

z95 = norminv(alpha95);
z99 = norminv(alpha99);

VaR95 = mu_p + z95 * sigma_p;
VaR99 = mu_p + z99 * sigma_p;

PortfolioValue = 1e6; % $1,000,000
DollarVaR95 = -VaR95 * PortfolioValue;
DollarVaR99 = -VaR99 * PortfolioValue;

portfolioReturns = returns * w;

histogram(portfolioReturns, 50)
hold on

x = linspace(min(portfolioReturns), max(portfolioReturns), 100);
y = normpdf(x, mu_p, sigma_p);

plot(x, y * length(portfolioReturns) * (max(portfolioReturns)-min(portfolioReturns))/50, ...
     'r', 'LineWidth',2)

title('Portfolio Returns with Normal Fit')

