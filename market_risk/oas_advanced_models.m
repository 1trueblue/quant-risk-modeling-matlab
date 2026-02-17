%% ADVANCED OAS MODELS IN MATLAB (WITH VISUALS)
% Sections 11–13:
% 11. Binomial Tree OAS (Non-Callable)
% 12. Monte Carlo OAS
% 13. Callable Bond OAS via Binomial Tree
%
% This file is fully standalone and version-proof.
%
% Author: <Your Name>

clc; clear; close all;

%% =========================================================
%% COMMON INPUTS
%% =========================================================

FaceValue = 100;
CouponRate = 0.06;
Coupon = FaceValue * CouponRate;

BaseRate = 0.05;
MarketPrice = 100;

%% =========================================================
%% SECTION 11: BINOMIAL TREE OAS (NON-CALLABLE)
%% =========================================================

dt = 1;
N  = 3;
r0 = 0.05;
sigma = 0.20;

u = exp(sigma * sqrt(dt));
d = 1/u;

r = zeros(N+1, N+1);
r(1,1) = r0;

for i = 2:N+1
    for j = 1:i
        r(j,i) = r0 * u^(i-j) * d^(j-1);
    end
end

PriceWithOAS = @(s) priceBondWithOAS(r, s, FaceValue, Coupon, dt, N);

OAS_tree = fzero(@(s) PriceWithOAS(s) - MarketPrice, 0.01);
fprintf("Binomial Tree OAS (Non-Callable): %.2f bps\n", OAS_tree*1e4);

%% --- Plot: Short-rate tree ---
figure;
for i = 1:N+1
    plot(0:i-1, r(1:i,i), 'o-'); hold on;
end
xlabel("Time Step");
ylabel("Short Rate");
title("Binomial Short-Rate Tree");
grid on;

%% --- Plot: Price vs Spread ---
Spreads = linspace(-0.02, 0.05, 50);
Prices_tree = arrayfun(PriceWithOAS, Spreads);

figure;
plot(Spreads*100, Prices_tree, 'LineWidth', 2);
xlabel("Spread (bps)");
ylabel("Bond Price");
title("Binomial Tree Price vs Spread (OAS Intuition)");
grid on;

%% =========================================================
%% SECTION 12: MONTE CARLO OAS
%% =========================================================

T = 36;   % months
CashFlows = zeros(T,1);
CashFlows(1:end-1) = Coupon/12;
CashFlows(end) = Coupon/12 + FaceValue;

dt_mc = 1/12;
NumPaths = 5000;
sigma_mc = 0.20;

Rates = zeros(T, NumPaths);
Rates(1,:) = BaseRate;

for t = 2:T
    Z = randn(1, NumPaths);
    Rates(t,:) = Rates(t-1,:) .* ...
        exp((-0.5*sigma_mc^2)*dt_mc + sigma_mc*sqrt(dt_mc).*Z);
end

MC_Price_With_OAS = @(s) mean( ...
    sum(CashFlows .* exp(-cumsum((Rates + s)*dt_mc,1)),1));

MC_OAS = fzero(@(s) MC_Price_With_OAS(s) - MarketPrice, 0.01);
fprintf("Monte Carlo OAS: %.2f bps\n", MC_OAS*1e4);

%% --- Plot: Sample interest-rate paths ---
figure;
plot(Rates(:,1:50));
xlabel("Month");
ylabel("Short Rate");
title("Monte Carlo Short-Rate Paths (Sample)");
grid on;

%% --- Plot: Monte Carlo price vs spread ---
Prices_MC = arrayfun(MC_Price_With_OAS, Spreads);

figure;
plot(Spreads*100, Prices_MC, 'LineWidth', 2);
xlabel("Spread (bps)");
ylabel("Bond Price");
title("Monte Carlo OAS: Price vs Spread");
grid on;

%% =========================================================
%% SECTION 13: CALLABLE BOND OAS (BINOMIAL TREE)
%% =========================================================

CallPrice = 100;
N_call = 4;
dt_call = 1;

r_call = zeros(N_call+1, N_call+1);
r_call(1,1) = r0;

for i = 2:N_call+1
    for j = 1:i
        r_call(j,i) = r0 * u^(i-j) * d^(j-1);
    end
end

CallablePriceWithOAS = @(s) priceCallableWithOAS( ...
    r_call, s, FaceValue, Coupon, CallPrice, dt_call, N_call);

MarketPriceCallable = CallablePriceWithOAS(0);
CallableOAS = fzero(@(s) ...
    CallablePriceWithOAS(s) - MarketPriceCallable, 0.01);

fprintf("Callable Bond OAS (Tree-Based): %.2f bps\n", CallableOAS*1e4);

%% --- Plot: Callable vs Non-Callable price ---
Prices_callable = arrayfun(CallablePriceWithOAS, Spreads);
Prices_noncall  = arrayfun(PriceWithOAS, Spreads);

figure;
plot(Spreads*100, Prices_noncall, 'LineWidth', 2); hold on;
plot(Spreads*100, Prices_callable, '--', 'LineWidth', 2);
xlabel("Spread (bps)");
ylabel("Bond Price");
title("Callable vs Non-Callable Bond (OAS Effect)");
legend("Non-Callable","Callable");
grid on;

%% =========================================================
%% LOCAL FUNCTIONS
%% =========================================================

function P = priceBondWithOAS(r, s, FaceValue, Coupon, dt, N)
    V = zeros(N+1, N+1);
    for j = 1:N+1
        V(j,N+1) = FaceValue + Coupon;
    end
    for i = N:-1:1
        for j = 1:i
            r_adj = r(j,i) + s;
            DF = exp(-r_adj * dt);
            V(j,i) = DF * 0.5 * ...
                (V(j,i+1) + V(j+1,i+1)) + Coupon;
        end
    end
    P = V(1,1);
end

function P = priceCallableWithOAS(r, s, FaceValue, Coupon, CallPrice, dt, N)
    V = zeros(N+1, N+1);
    for j = 1:N+1
        V(j,N+1) = FaceValue + Coupon;
    end
    for i = N:-1:1
        for j = 1:i
            r_adj = r(j,i) + s;
            DF = exp(-r_adj * dt);
            HoldValue = DF * 0.5 * ...
                (V(j,i+1) + V(j+1,i+1)) + Coupon;
            V(j,i) = min(HoldValue, CallPrice);
        end
    end
    P = V(1,1);
end
