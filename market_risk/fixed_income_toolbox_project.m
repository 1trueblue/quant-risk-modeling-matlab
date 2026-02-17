%% FIXED INCOME TOOLBOX CAPSTONE PROJECT (MANUAL, ROBUST)
% Integrated Yield Curve Construction, MBS Analytics,
% Effective Duration, OAS, and Interest Rate Derivatives
%
% NOTE:
% This implementation avoids buggy MBS toolbox functions
% and builds all MBS analytics from first principles.
%
% Author: <Your Name>

clc; clear; close all;

%% =========================================================
%% 1. DATE SETUP AND MARKET DATA
%% =========================================================

Settle = datenum('01-Jan-2024');

MaturitiesYrs = [0.25 0.5 1 2 5 10];
MaturityDates = Settle + round(365 * MaturitiesYrs);
MarketYields  = [0.045 0.047 0.048 0.050 0.052 0.055];

%% =========================================================
%% 2. ZERO CURVE (SIMPLIFIED)
%% =========================================================

ZeroRates = MarketYields;

IRCurve = IRDataCurve( ...
    'zero', Settle, MaturityDates, ZeroRates, ...
    'Compounding', -1);

%% =========================================================
%% 3. FORWARD RATES (ROBUST, MANUAL)
%% =========================================================

DF_Z = getDiscountFactors(IRCurve, MaturityDates);
YearFracs = diff(MaturityDates) / 365;

ForwardRates = zeros(length(YearFracs),1);
for i = 1:length(YearFracs)
    ForwardRates(i) = (DF_Z(i) / DF_Z(i+1))^(1/YearFracs(i)) - 1;
end

disp("Forward Rates (annualized):")
disp(ForwardRates)

%% =========================================================
%% 4. NELSON–SIEGEL TERM STRUCTURE (MANUAL)
%% =========================================================

Tau = (MaturityDates - Settle) / 365;

NSFun = @(b,t) ...
    b(1) + ...
    b(2).*((1-exp(-t./b(4)))./(t./b(4))) + ...
    b(3).*(((1-exp(-t./b(4)))./(t./b(4))) - exp(-t./b(4)));

b0 = [0.05 -0.02 0.02 2];
NSParams = lsqcurvefit(NSFun, b0, Tau, ZeroRates);

TauFine = linspace(min(Tau), max(Tau), 100);
NSZeroRates = NSFun(NSParams, TauFine);

figure;
plot(Tau, ZeroRates, 'o', TauFine, NSZeroRates, 'LineWidth', 2)
xlabel("Maturity (Years)")
ylabel("Zero Rate")
title("Nelson–Siegel Yield Curve")
legend("Market","NS Fit")
grid on

%% =========================================================
%% 5. MBS CASH FLOWS WITH PREPAYMENT (MANUAL)
%% =========================================================

LoanAmount = 100;
AnnualRate = 0.06;
TermYears  = 30;
NumMonths  = TermYears * 12;
rm = AnnualRate / 12;

PMT = LoanAmount * (rm*(1+rm)^NumMonths) / ((1+rm)^NumMonths - 1);

PSA = 100;
CPR_max = 0.06;

CPR = CPR_max * PSA/100 .* min((1:NumMonths)'/30,1);
SMM = 1 - (1 - CPR).^(1/12);

Balance = zeros(NumMonths+1,1);
Balance(1) = LoanAmount;

CashFlows = zeros(NumMonths,1);

for t = 1:NumMonths
    Interest  = Balance(t) * rm;
    SchedPrin = PMT - Interest;
    Prepay    = SMM(t) * max(Balance(t) - SchedPrin, 0);
    
    Balance(t+1) = Balance(t) - SchedPrin - Prepay;
    CashFlows(t) = Interest + SchedPrin + Prepay;
end

lastCF = find(Balance <= 0, 1);
CashFlows = CashFlows(1:lastCF);

%% =========================================================
%% 6. MBS PRICE AND EFFECTIVE DURATION (MANUAL)
%% =========================================================

BaseRate = 0.05;

DF = (1 + BaseRate/12).^(-(1:length(CashFlows))');
Price = sum(CashFlows .* DF);

Shock = 0.0001;  % 1 bp

DF_up   = (1 + (BaseRate+Shock)/12).^(-(1:length(CashFlows))');
DF_down = (1 + (BaseRate-Shock)/12).^(-(1:length(CashFlows))');

PV_up   = sum(CashFlows .* DF_up);
PV_down = sum(CashFlows .* DF_down);

EffDuration = (PV_down - PV_up) / (2 * Price * Shock);

fprintf("MBS Price: %.2f\n", Price)
fprintf("Effective Duration: %.2f years\n", EffDuration)

%% =========================================================
%% 7. OPTION-ADJUSTED SPREAD (MANUAL ROOT-FINDING)
%% =========================================================

MarketPrice = Price;  % assume model price equals market price

OAS_fun = @(s) sum(CashFlows .* ...
    (1 + (BaseRate+s)/12).^(-(1:length(CashFlows))')) - MarketPrice;

OAS = fzero(OAS_fun, 0.01);

fprintf("Option-Adjusted Spread (OAS): %.2f bps\n", OAS*1e4)

%% =========================================================
%% 8. NEGATIVE CONVEXITY
%% =========================================================

TestRates = linspace(0.03, 0.08, 30);
Prices = zeros(size(TestRates));

for i = 1:length(TestRates)
    DF = (1 + TestRates(i)/12).^(-(1:length(CashFlows))');
    Prices(i) = sum(CashFlows .* DF);
end

figure;
plot(TestRates*100, Prices, 'LineWidth', 2)
xlabel("Interest Rate (%)")
ylabel("MBS Price")
title("Negative Convexity of MBS")
grid on

%% =========================================================
%% 9. OAS INTUITION: PRICE VS SPREAD
%% =========================================================

Spreads = linspace(0, 0.03, 30);
OAS_Prices = zeros(size(Spreads));

for i = 1:length(Spreads)
    DF = (1 + (BaseRate + Spreads(i))/12).^(-(1:length(CashFlows))');
    OAS_Prices(i) = sum(CashFlows .* DF);
end

figure;
plot(Spreads*100, OAS_Prices, 'LineWidth', 2)
xlabel("Spread (bps)")
ylabel("MBS Price")
title("OAS Intuition")
grid on


%% =========================================================
%% 10. BINOMIAL INTEREST-RATE TREE (CALLABLE BOND)
%% =========================================================

% --- Model parameters ---
r0    = 0.05;     % Initial short rate (5%)
sigma = 0.20;     % Interest rate volatility
dt    = 1;        % Time step = 1 year
N     = 3;        % 3-year tree

FaceValue = 100;
CouponRate = 0.06;
Coupon = FaceValue * CouponRate;

CallPrice = 100;  % Callable at par

% --- Build short-rate tree ---
r = zeros(N+1, N+1);
r(1,1) = r0;

u = exp(sigma * sqrt(dt));
d = 1/u;

for i = 2:N+1
    for j = 1:i
        r(j,i) = r0 * u^(i-j) * d^(j-1);
    end
end

% --- Initialize bond values at maturity ---
BondValue = zeros(N+1, N+1);

for j = 1:N+1
    BondValue(j,N+1) = FaceValue + Coupon;
end

% --- Backward induction ---
for i = N:-1:1
    for j = 1:i
        Disc = exp(-r(j,i) * dt);
        HoldValue = Disc * 0.5 * ...
            (BondValue(j,i+1) + BondValue(j+1,i+1));
        
        % Callable feature
        BondValue(j,i) = min(HoldValue + Coupon, CallPrice);
    end
end

CallableBondPrice = BondValue(1,1);

fprintf("Callable Bond Price (Binomial Tree): %.2f\n", CallableBondPrice)



%Additional part
%Plotting the short rate tree
figure;
for i = 1:N+1
    plot(0:i-1, r(1:i,i), 'o-'); hold on;
end
xlabel("Time Step")
ylabel("Short Rate")
title("Binomial Short-Rate Tree")
grid on

%Compare callable vs non-callable bond
NonCallableValue = zeros(N+1, N+1);

for j = 1:N+1
    NonCallableValue(j,N+1) = FaceValue + Coupon;
end

for i = N:-1:1
    for j = 1:i
        Disc = exp(-r(j,i) * dt);
        NonCallableValue(j,i) = ...
            Disc * 0.5 * ...
            (NonCallableValue(j,i+1) + NonCallableValue(j+1,i+1)) + Coupon;
    end
end

fprintf("Non-Callable Bond Price: %.2f\n", NonCallableValue(1,1))


%% =========================================================
%% END OF PROJECT
%% =========================================================
