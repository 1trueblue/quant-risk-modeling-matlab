%% PROJECT: Interest-Rate Risk and MBS Valuation
% Author: <Your Name>
% Description:
% Models mortgage cash flows, prepayment behavior,
% pricing, duration, and OAS intuition


%% PROJECT SUMMARY: MBS Valuation, Duration, Negative Convexity, and OAS Intuition
%
% This script implements a simplified fixed-income model to analyze a
% Mortgage-Backed Security (MBS) using core concepts from bond mathematics
% and prepayment theory.
%
% 1. Mortgage Cash Flow Modeling:
%    - A fixed-rate mortgage is modeled as an annuity loan with a constant
%      monthly payment determined using time-value-of-money principles.
%    - Each monthly payment is decomposed into interest and scheduled
%      principal repayment based on the outstanding loan balance.
%
% 2. Amortization Schedule (No Prepayment Baseline):
%    - A month-by-month amortization schedule is constructed assuming no
%      borrower prepayment.
%    - This provides deterministic baseline cash flows against which
%      prepayment effects can be studied.
%
% 3. Prepayment Modeling (PSA Framework):
%    - Borrower prepayment behavior is introduced using a PSA-style model,
%      where the Conditional Prepayment Rate (CPR) ramps up linearly in the
%      early life of the mortgage and then stabilizes.
%    - CPR is converted to Single Monthly Mortality (SMM) to obtain monthly
%      prepayment rates consistent with the annual CPR assumption.
%    - Monthly prepayments are calculated as a fraction of the remaining
%      loan balance after scheduled principal repayment.
%
% 4. MBS Cash Flows:
%    - Total cash flows to the investor consist of interest payments,
%      scheduled principal, and prepaid principal.
%    - Prepayment alters the timing of principal return, making MBS cash
%      flows path-dependent and interest-rate sensitive.
%
% 5. Pricing via Discounting:
%    - MBS price is computed as the present value of expected cash flows
%      discounted using a flat yield curve.
%
% 6. Effective Duration (Rate Shock Method):
%    - Standard duration formulas are not applicable due to rate-dependent
%      cash flows.
%    - Effective duration is computed numerically by re-pricing the MBS
%      under small upward and downward shocks to the discount rate.
%    - This captures the true interest-rate sensitivity of the MBS.
%
% 7. Negative Convexity Analysis:
%    - MBS prices are computed across a range of interest rates.
%    - The resulting price–yield curve demonstrates negative convexity:
%      price appreciation is limited when rates fall (due to prepayment),
%      while price declines are amplified when rates rise (due to extension
%      risk).
%
% 8. Option-Adjusted Spread (OAS) Intuition:
%    - A constant spread is added to the benchmark discount rate to reflect
%      additional return demanded by investors for embedded prepayment risk.
%    - Prices are computed across different spread levels to show how higher
%      required compensation leads to lower market prices.
%    - This provides intuition for OAS as the spread that reconciles modeled
%      prices with observed market prices after accounting for optionality.
%
% Overall, this script demonstrates how borrower refinancing behavior,
% prepayment risk, and interest-rate movements fundamentally alter MBS
% valuation, duration stability, and required investor compensation.
%

clc; clear; close all;

%% 1. BASIC MORTGAGE PARAMETERS

LoanAmount   = 1000000;     % Principal (₹ / $)
AnnualRate   = 0.06;          % Mortgage rate (6%)
MonthlyRate  = AnnualRate/12;
MaturityYrs  = 30;
NumMonths    = MaturityYrs * 12;

%% 2. MONTHLY MORTGAGE PAYMENT (FIXED)

MonthlyPayment = LoanAmount * (MonthlyRate * (1+MonthlyRate)^NumMonths) / ((1+MonthlyRate)^NumMonths - 1);

%% 3. BUILD SCHEDULE WITHOUT PREPAYMENT

Balance   = zeros(NumMonths,1);
Interest  = zeros(NumMonths,1);
Principal = zeros(NumMonths,1);

Balance(1) = LoanAmount;

for t = 1:NumMonths
    Interest(t)  = Balance(t) * MonthlyRate;
    Principal(t) = MonthlyPayment - Interest(t);
    if t < NumMonths
        Balance(t+1) = Balance(t) - Principal(t);
    end
end

%% 4. INTRODUCE PREPAYMENT (PSA-LIKE SIMPLIFICATION)

% CPR ramps up for first 30 months, then constant
CPR = zeros(NumMonths,1);

for t = 1:NumMonths
    if t <= 30
        CPR(t) = 0.06 * (t/30);   % Ramp to 6%
    else
        CPR(t) = 0.06;
    end
end

% Convert CPR to SMM
SMM = 1 - (1 - CPR).^(1/12);

Prepayment = zeros(NumMonths,1);

for t = 1:NumMonths
    Prepayment(t) = (Balance(t) - Principal(t)) * SMM(t);
end

%% 5. TOTAL CASH FLOWS TO INVESTOR

TotalPrincipal = Principal + Prepayment;
CashFlows = Interest + TotalPrincipal;

%% 6. DISCOUNTING USING FLAT YIELD CURVE

DiscountRate = 0.05;     % Flat 5% yield
MonthlyDisc  = DiscountRate / 12;

DiscountFactors = (1 + MonthlyDisc).^(-(1:NumMonths)');
PV = sum(CashFlows .* DiscountFactors);

fprintf("MBS Price = %.2f\n", PV);

%% 7. EFFECTIVE DURATION (RATE SHOCK METHOD)

Shock = 0.0001;  % 1 bp

PV_up   = sum(CashFlows .* (1 + (DiscountRate+Shock)/12).^(-(1:NumMonths)'));
PV_down = sum(CashFlows .* (1 + (DiscountRate-Shock)/12).^(-(1:NumMonths)'));

Duration = (PV_down - PV_up) / (2 * PV * Shock);

fprintf("Effective Duration = %.2f years\n", Duration);

%% 8. NEGATIVE CONVEXITY DEMONSTRATION

Rates = linspace(0.03,0.08,20);
Prices = zeros(length(Rates),1);

for i = 1:length(Rates)
    disc = (1 + Rates(i)/12).^(-(1:NumMonths)');
    Prices(i) = sum(CashFlows .* disc);
end

figure;
plot(Rates*100, Prices, 'LineWidth',2)
xlabel("Interest Rate (%)")
ylabel("MBS Price")
title("Negative Convexity of MBS")
grid on

%% 9. OAS INTUITION (SPREAD ADJUSTMENT)

Spreads = linspace(0,0.03,20);
OAS_Prices = zeros(length(Spreads),1);

for i = 1:length(Spreads)
    disc = (1 + (DiscountRate + Spreads(i))/12).^(-(1:NumMonths)');
    OAS_Prices(i) = sum(CashFlows .* disc);
end

figure;
plot(Spreads*100, OAS_Prices, 'LineWidth',2)
xlabel("Spread (%)")
ylabel("Price")
title("Option-Adjusted Spread (OAS) Intuition")
grid on
