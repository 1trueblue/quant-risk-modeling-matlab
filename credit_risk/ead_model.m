%% EXPOSURE AT DEFAULT (EAD) MODEL
% Predicting credit conversion factor and exposure at default
% Dataset: 800 credit facilities

%% 1. IMPORT DATA
clear; clc; close all;

data = readtable('ead_model_dataset.csv');

fprintf('========================================\n');
fprintf('EAD MODEL - DATASET OVERVIEW\n');
fprintf('========================================\n');
fprintf('Total facilities: %d\n', height(data));
fprintf('Mean Utilization: %.2f%%\n', mean(data.Utilization) * 100);
fprintf('Mean CCF: %.2f%%\n', mean(data.CCF) * 100);
fprintf('Mean EAD%%: %.2f%%\n\n', mean(data.EAD_Percent) * 100);

disp('Sample Data:');
disp(head(data, 5));

%% 2. EXPLORATORY ANALYSIS

fprintf('\n========================================\n');
fprintf('EAD METRICS BY FACILITY TYPE\n');
fprintf('========================================\n');
facility_summary = groupsummary(data, 'FacilityType', 'mean', {'Utilization', 'CCF', 'EAD_Percent'});
facility_summary.mean_Utilization = facility_summary.mean_Utilization * 100;
facility_summary.mean_CCF = facility_summary.mean_CCF * 100;
facility_summary.mean_EAD_Percent = facility_summary.mean_EAD_Percent * 100;
facility_summary.Properties.VariableNames = {'FacilityType', 'Count', ...
    'Avg_Utilization_Pct', 'Avg_CCF_Pct', 'Avg_EAD_Pct'};
disp(facility_summary);

%% 3. VISUALIZATIONS

figure('Name', 'EAD Analysis', 'Position', [100 100 1400 900]);

% Plot 1: CCF Distribution
subplot(3,3,1);
histogram(data.CCF * 100, 30, 'FaceColor', [0.3 0.6 0.3]);
xlabel('Credit Conversion Factor (%)');
ylabel('Frequency');
title('CCF Distribution');
grid on;

% Plot 2: CCF by Facility Type
subplot(3,3,2);
boxplot(data.CCF * 100, data.FacilityType);
ylabel('CCF (%)');
title('CCF by Facility Type');
xtickangle(45);
grid on;

% Plot 3: EAD% Distribution
subplot(3,3,3);
histogram(data.EAD_Percent * 100, 30, 'FaceColor', [0.6 0.3 0.6]);
xlabel('EAD (% of Limit)');
ylabel('Frequency');
title('EAD Percent Distribution');
grid on;

% Plot 4: Utilization vs CCF
subplot(3,3,4);
scatter(data.Utilization * 100, data.CCF * 100, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Current Utilization (%)');
ylabel('CCF (%)');
title('Utilization vs CCF');
grid on;

% Plot 5: Credit Score vs CCF
subplot(3,3,5);
scatter(data.CreditScore, data.CCF * 100, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Credit Score');
ylabel('CCF (%)');
title('Credit Score vs CCF');
grid on;

% Plot 6: Maturity vs CCF
subplot(3,3,6);
scatter(data.MaturityMonths, data.CCF * 100, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Maturity (Months)');
ylabel('CCF (%)');
title('Maturity vs CCF');
grid on;

% Plot 7: Industry Stress Impact
subplot(3,3,7);
stressed = data(data.IndustryStress == 1, :);
non_stressed = data(data.IndustryStress == 0, :);
bar([mean(non_stressed.CCF)*100, mean(stressed.CCF)*100]);
set(gca, 'XTickLabel', {'No Stress', 'Industry Stress'});
ylabel('Mean CCF (%)');
title('Industry Stress Impact on CCF');
grid on;

% Plot 8: Current Balance vs EAD
subplot(3,3,8);
scatter(data.CurrentBalance_M, data.EAD_M, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Current Balance ($M)');
ylabel('EAD ($M)');
title('Current Balance vs EAD');
grid on;
hold on;
plot([0 max(data.CurrentBalance_M)], [0 max(data.CurrentBalance_M)], 'r--', 'LineWidth', 2);
legend('Data', 'No Drawdown Line', 'Location', 'northwest');

% Plot 9: Unused Limit vs Drawdown
subplot(3,3,9);
drawdown = data.EAD_M - data.CurrentBalance_M;
scatter(data.UnusedLimit_M, drawdown, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Unused Limit ($M)');
ylabel('Additional Drawdown at Default ($M)');
title('Unused Limit vs Drawdown');
grid on;

%% 4. PREPARE DATA FOR MODELING

% Convert categorical to dummies
facility_dummies = dummyvar(categorical(data.FacilityType));

% Feature matrix (predicting CCF)
X = [data.Utilization, ...
     data.CreditScore / 1000, ... % Normalize
     data.MaturityMonths / 100, ... % Normalize
     data.IndustryStress, ...
     facility_dummies];

% Target: Credit Conversion Factor
y = data.CCF;

%% 5. TRAIN-TEST SPLIT

rng(42);
cv = cvpartition(length(y), 'HoldOut', 0.25);

X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

fprintf('\n========================================\n');
fprintf('TRAIN-TEST SPLIT\n');
fprintf('========================================\n');
fprintf('Training samples: %d\n', length(y_train));
fprintf('Test samples: %d\n', length(y_test));

%% 6. LINEAR REGRESSION MODEL

fprintf('\n========================================\n');
fprintf('TRAINING CCF PREDICTION MODEL\n');
fprintf('========================================\n');

% Fit model
mdl = fitlm(X_train, y_train);

% Display model
disp(mdl);

% Get R-squared
r_squared = mdl.Rsquared.Ordinary;
fprintf('\nModel R-squared: %.4f\n', r_squared);

%% 7. PREDICTIONS AND EVALUATION

% Predict on test set
y_pred = predict(mdl, X_test);

% Clip predictions to [0, 1] range
y_pred = max(0, min(1, y_pred));

% Calculate metrics
mae = mean(abs(y_test - y_pred));
rmse = sqrt(mean((y_test - y_pred).^2));
mape = mean(abs((y_test - y_pred) ./ y_test)) * 100;

fprintf('\n========================================\n');
fprintf('MODEL PERFORMANCE - TEST SET\n');
fprintf('========================================\n');
fprintf('MAE (CCF): %.4f (%.2f%%)\n', mae, mae * 100);
fprintf('RMSE (CCF): %.4f (%.2f%%)\n', rmse, rmse * 100);
fprintf('MAPE: %.2f%%\n', mape);

%% 8. CALCULATE EAD FROM PREDICTED CCF

% For test set, calculate predicted EAD
test_indices = test(cv);
current_balance_test = data.CurrentBalance_M(test_indices);
unused_limit_test = data.UnusedLimit_M(test_indices);

% Predicted EAD
ead_pred = current_balance_test + (unused_limit_test .* y_pred);
ead_actual = data.EAD_M(test_indices);

% EAD prediction error
ead_mae = mean(abs(ead_actual - ead_pred));
ead_mape = mean(abs((ead_actual - ead_pred) ./ ead_actual)) * 100;

fprintf('\nEAD PREDICTION PERFORMANCE:\n');
fprintf('MAE (EAD $M): $%.2fM\n', ead_mae);
fprintf('MAPE (EAD): %.2f%%\n', ead_mape);

%% 9. VISUALIZE PREDICTIONS

figure('Name', 'EAD Model Results', 'Position', [100 100 1400 500]);

% Plot 1: Actual vs Predicted CCF
subplot(1,3,1);
scatter(y_test * 100, y_pred * 100, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([0 100], [0 100], 'r--', 'LineWidth', 2);
xlabel('Actual CCF (%)');
ylabel('Predicted CCF (%)');
title('CCF: Actual vs Predicted');
grid on;
legend('Predictions', 'Perfect Fit', 'Location', 'northwest');
axis equal;
xlim([0 100]);
ylim([0 100]);

% Plot 2: CCF Residuals
subplot(1,3,2);
residuals = y_test - y_pred;
histogram(residuals * 100, 30, 'FaceColor', [0.8 0.2 0.2]);
xlabel('CCF Prediction Error (%)');
ylabel('Frequency');
title('CCF Prediction Errors');
grid on;
xline(0, 'b--', 'LineWidth', 2);

% Plot 3: Actual vs Predicted EAD
subplot(1,3,3);
scatter(ead_actual, ead_pred, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
max_ead = max([ead_actual; ead_pred]);
plot([0 max_ead], [0 max_ead], 'r--', 'LineWidth', 2);
xlabel('Actual EAD ($M)');
ylabel('Predicted EAD ($M)');
title('EAD: Actual vs Predicted');
grid on;
legend('Predictions', 'Perfect Fit', 'Location', 'northwest');

%% 10. FEATURE IMPORTANCE

fprintf('\n========================================\n');
fprintf('FEATURE IMPORTANCE\n');
fprintf('========================================\n');

% Get coefficients
coefs = mdl.Coefficients.Estimate(2:end);
pvals = mdl.Coefficients.pValue(2:end);

% Feature names
feature_names = {'Utilization', 'CreditScore', 'Maturity', 'IndustryStress', ...
    'Facility_1', 'Facility_2', 'Facility_3', 'Facility_4'};

% Sort by absolute coefficient
[~, sort_idx] = sort(abs(coefs), 'descend');

fprintf('%-25s %12s %12s %12s\n', 'Feature', 'Coefficient', 't-stat', 'p-value');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:length(sort_idx)
    idx = sort_idx(i);
    fprintf('%-25s %12.4f %12.4f %12.6f', feature_names{idx}, ...
        coefs(idx), mdl.Coefficients.tStat(idx+1), pvals(idx));
    
    if pvals(idx) < 0.001
        fprintf(' ***\n');
    elseif pvals(idx) < 0.01
        fprintf(' **\n');
    elseif pvals(idx) < 0.05
        fprintf(' *\n');
    else
        fprintf('\n');
    end
end

%% 11. SCENARIO ANALYSIS

fprintf('\n========================================\n');
fprintf('SCENARIO ANALYSIS\n');
fprintf('========================================\n');

% Define scenarios (Utilization, CreditScore/1000, Maturity/100, IndustryStress, Facility dummies)
scenarios = {
    'Revolver - Low Util, Good Credit', [0.30, 0.750, 0.30, 0, 0, 0, 1, 0];
    'Revolver - High Util, Poor Credit', [0.80, 0.450, 0.30, 1, 0, 0, 1, 0];
    'Term Loan - Fully Drawn', [0.98, 0.650, 0.60, 0, 0, 0, 0, 1];
    'Credit Card - Moderate', [0.50, 0.600, 0.12, 0, 1, 0, 0, 0];
};

fprintf('%-40s %12s %12s\n', 'Scenario', 'CCF', 'EAD%');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:size(scenarios, 1)
    scenario_name = scenarios{i, 1};
    scenario_X = scenarios{i, 2};
    
    predicted_ccf = predict(mdl, scenario_X);
    predicted_ccf = max(0, min(1, predicted_ccf));
    
    % Calculate EAD% (assuming current utilization)
    current_util = scenario_X(1);
    ead_pct = current_util + ((1 - current_util) * predicted_ccf);
    
    fprintf('%-40s %11.2f%% %11.2f%%\n', scenario_name, ...
        predicted_ccf * 100, ead_pct * 100);
end

%% 12. SUMMARY AND INSIGHTS

fprintf('\n========================================\n');
fprintf('KEY INSIGHTS\n');
fprintf('========================================\n');

fprintf('\n1. MODEL PERFORMANCE:\n');
fprintf('   - R-squared: %.2f%% of CCF variance explained\n', r_squared * 100);
fprintf('   - Average CCF prediction error: %.2f%%\n', mae * 100);
fprintf('   - Average EAD prediction error: $%.2fM\n', ead_mae);

fprintf('\n2. CCF BY FACILITY TYPE:\n');
for i = 1:height(facility_summary)
    fprintf('   - %s: %.0f%% CCF\n', facility_summary.FacilityType{i}, ...
        facility_summary.Avg_CCF_Pct(i));
end

fprintf('\n3. KEY DRIVERS OF DRAWDOWN:\n');
fprintf('   - Lower credit quality → Higher CCF (desperate drawdown)\n');
fprintf('   - Industry stress → Higher CCF\n');
fprintf('   - Current utilization → Complex relationship\n');

fprintf('\n4. REGULATORY CAPITAL IMPLICATIONS:\n');
avg_ccf = mean(data.CCF);
fprintf('   - Portfolio average CCF: %.0f%%\n', avg_ccf * 100);
fprintf('   - Unused commitments should carry %.0f%% weight\n', avg_ccf * 100);

fprintf('\n✓ EAD Model Complete!\n');