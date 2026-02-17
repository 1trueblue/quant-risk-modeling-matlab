%% MERTON STRUCTURAL MODEL – PORTFOLIO VERSION (CORRECTED)

clear; clc; close all;

%% ==============================
%% STEP 1 — Load Data
%% ==============================

data = readtable('credit_risk_portfolio_data.csv');

tickers = unique(data.Ticker);
nFirms = length(tickers);

T = 1;   % 1-year horizon

%% ==============================
%% Preallocate Results Table
%% ==============================

results = table( ...
    strings(nFirms,1), ...
    zeros(nFirms,1), ...
    zeros(nFirms,1), ...
    zeros(nFirms,1), ...
    zeros(nFirms,1), ...
    zeros(nFirms,1), ...
    'VariableNames', ...
    {'Ticker','AssetValue','AssetVolatility','DebtBarrier','DistanceToDefault','PD_1Y'});

%% ==============================
%% STEP 2 — Loop Over Firms
%% ==============================

for i = 1:nFirms

    firm = tickers{i};
    firmData = data(strcmp(data.Ticker, firm), :);

    % Use most recent observation
    row = firmData(end,:);

    % Equity value
    E = row.Market_Cap;

    % ---- REALISTIC DEFAULT BARRIER ----
    % If you only have TotalDebt:
    D = row.Total_Debt;

    % If later you add ShortTermDebt & LongTermDebt:
    % D = row.ShortTermDebt + 0.5 * row.LongTermDebt;

    r = row.Risk_Free_Rate;
    sigma_E = row.Equity_Volatility;

    %% Solve Merton Nonlinear System

    mertonSystem = @(x) mertonEquations(x, E, sigma_E, D, r, T);

    initGuess = [E + D; sigma_E];

    options = optimoptions('fsolve','Display','off','FunctionTolerance',1e-8);
    solution = fsolve(mertonSystem, initGuess, options);

    A = solution(1);
    sigma_A = solution(2);

    %% Distance to Default

    d2 = (log(A/D) + (r - 0.5*sigma_A^2)*T) / (sigma_A*sqrt(T));
    DD = d2;

    %% Probability of Default

    PD = normcdf(-d2);

    %% Store Results

    results.Ticker(i) = string(firm);
    results.AssetValue(i) = A;
    results.AssetVolatility(i) = sigma_A;
    results.DebtBarrier(i) = D;
    results.DistanceToDefault(i) = DD;
    results.PD_1Y(i) = PD;

end

%% ==============================
%% STEP 3 — Display Results
%% ==============================

disp(' ');
disp('===== MERTON MODEL PORTFOLIO RESULTS =====');
disp(results);

%% ==============================
%% STEP 4 — Sort by Risk
%% ==============================

resultsSorted = sortrows(results,'PD_1Y','descend');

disp(' ');
disp('===== FIRMS SORTED BY DEFAULT RISK =====');
disp(resultsSorted);

%% ==============================
%% Merton Equation System
%% ==============================

function F = mertonEquations(x, E, sigma_E, D, r, T)

A = x(1);
sigma_A = x(2);

d1 = (log(A/D) + (r + 0.5*sigma_A^2)*T) / (sigma_A*sqrt(T));
d2 = d1 - sigma_A*sqrt(T);

eq1 = A*normcdf(d1) - D*exp(-r*T)*normcdf(d2) - E;
eq2 = (A/E)*normcdf(d1)*sigma_A - sigma_E;

F = [eq1; eq2];

end
