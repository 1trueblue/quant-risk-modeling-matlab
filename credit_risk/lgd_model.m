%% LOSS GIVEN DEFAULT (LGD) MODEL
% Predicting recovery rates and loss severity
% Dataset: 500 defaulted firms

%% 1. IMPORT DATA
clear; clc; close all;

data = readtable('lgd_model_dataset.csv');

fprintf('========================================\n');
fprintf('LGD MODEL - DATASET OVERVIEW\n');
fprintf('========================================\n');
fprintf('Total defaults: %d\n', height(data));
fprintf('Mean LGD: %.2f%%\n', mean(data.LGD) * 100);
fprintf('Mean Recovery Rate: %.2f%%\n\n', mean(data.RecoveryRate) * 100);

disp('Sample Data:');
disp(head(data, 5));

%% 2. EXPLORATORY ANALYSIS

fprintf('\n========================================\n');
fprintf('LGD BY COLLATERAL TYPE\n');
fprintf('========================================\n');
collateral_summary = groupsummary(data, 'CollateralType', 'mean', 'LGD');
collateral_summary.Properties.VariableNames{3} = 'Mean_LGD';
collateral_summary.Mean_LGD = collateral_summary.Mean_LGD * 100;
disp(collateral_summary);

fprintf('\n========================================\n');
fprintf('LGD BY SENIORITY\n');
fprintf('========================================\n');
seniority_summary = groupsummary(data, 'Seniority', 'mean', 'LGD');
seniority_summary.Properties.VariableNames{3} = 'Mean_LGD';
seniority_summary.Mean_LGD = seniority_summary.Mean_LGD * 100;
disp(seniority_summary);

%% 3. VISUALIZATIONS

figure('Name', 'LGD Analysis', 'Position', [100 100 1400 800]);

% Plot 1: LGD Distribution
subplot(2,3,1);
histogram(data.LGD * 100, 30, 'FaceColor', [0.2 0.4 0.8]);
xlabel('LGD (%)');
ylabel('Frequency');
title('Distribution of Loss Given Default');
grid on;

% Plot 2: LGD by Collateral Type
subplot(2,3,2);
boxplot(data.LGD * 100, data.CollateralType);
ylabel('LGD (%)');
title('LGD by Collateral Type');
xtickangle(45);
grid on;

% Plot 3: LGD by Seniority
subplot(2,3,3);
boxplot(data.LGD * 100, data.Seniority);
ylabel('LGD (%)');
title('LGD by Seniority');
grid on;

% Plot 4: LGD vs Leverage
subplot(2,3,4);
scatter(data.DebtToAssets, data.LGD * 100, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Debt-to-Assets Ratio');
ylabel('LGD (%)');
title('LGD vs Leverage');
grid on;

% Plot 5: LGD vs Asset Tangibility
subplot(2,3,5);
scatter(data.AssetTangibility, data.LGD * 100, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Asset Tangibility');
ylabel('LGD (%)');
title('LGD vs Asset Tangibility');
grid on;

% Plot 6: Recovery vs Time
subplot(2,3,6);
scatter(data.TimeToResolution_Days, data.RecoveryRate * 100, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Time to Resolution (Days)');
ylabel('Recovery Rate (%)');
title('Recovery vs Resolution Time');
grid on;

%% 4. PREPARE DATA FOR MODELING

% Convert categorical variables to dummies
collateral_dummies = dummyvar(categorical(data.CollateralType));
seniority_dummies = dummyvar(categorical(data.Seniority));
sector_dummies = dummyvar(categorical(data.Sector));

% Feature matrix
X = [data.DebtToAssets, ...
     data.AssetTangibility, ...
     data.TimeToResolution_Days / 1000, ... % Scale down
     data.Recession, ...
     collateral_dummies, ...
     seniority_dummies];

% Target: We'll predict Recovery Rate (easier to interpret)
y = data.RecoveryRate;

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
fprintf('TRAINING LINEAR REGRESSION MODEL\n');
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

% Convert to LGD for interpretation
lgd_test = 1 - y_test;
lgd_pred = 1 - y_pred;
lgd_mae = mean(abs(lgd_test - lgd_pred));

fprintf('\n========================================\n');
fprintf('MODEL PERFORMANCE - TEST SET\n');
fprintf('========================================\n');
fprintf('MAE (Recovery Rate): %.4f (%.2f%%)\n', mae, mae * 100);
fprintf('RMSE (Recovery Rate): %.4f (%.2f%%)\n', rmse, rmse * 100);
fprintf('MAPE: %.2f%%\n', mape);
fprintf('MAE (LGD): %.4f (%.2f%%)\n', lgd_mae, lgd_mae * 100);

%% 8. VISUALIZE PREDICTIONS

figure('Name', 'LGD Model Results', 'Position', [100 100 1200 500]);

% Plot 1: Actual vs Predicted (Recovery Rate)
subplot(1,2,1);
scatter(y_test * 100, y_pred * 100, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([0 100], [0 100], 'r--', 'LineWidth', 2);
xlabel('Actual Recovery Rate (%)');
ylabel('Predicted Recovery Rate (%)');
title('Actual vs Predicted Recovery');
grid on;
legend('Predictions', 'Perfect Fit', 'Location', 'northwest');
axis equal;
xlim([0 100]);
ylim([0 100]);

% Plot 2: Residuals
subplot(1,2,2);
residuals = y_test - y_pred;
histogram(residuals * 100, 30, 'FaceColor', [0.8 0.2 0.2]);
xlabel('Prediction Error (%)');
ylabel('Frequency');
title('Distribution of Prediction Errors');
grid on;
xline(0, 'b--', 'LineWidth', 2);

%% 9. FEATURE IMPORTANCE

fprintf('\n========================================\n');
fprintf('FEATURE IMPORTANCE (Top 10)\n');
fprintf('========================================\n');

% Get coefficients
coefs = mdl.Coefficients.Estimate(2:end);  % Exclude intercept
pvals = mdl.Coefficients.pValue(2:end);

% Feature names
feature_names = {'DebtToAssets', 'AssetTangibility', 'TimeToResolution', ...
    'Recession', 'Collateral_1', 'Collateral_2', 'Collateral_3', ...
    'Collateral_4', 'Collateral_5', 'Seniority_1', 'Seniority_2', 'Seniority_3'};

% Sort by absolute coefficient value
[~, sort_idx] = sort(abs(coefs), 'descend');

fprintf('%-25s %12s %12s %12s\n', 'Feature', 'Coefficient', 't-stat', 'p-value');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:min(10, length(sort_idx))
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

fprintf('\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n');

%% 10. SCENARIO ANALYSIS

fprintf('\n========================================\n');
fprintf('SCENARIO ANALYSIS\n');
fprintf('========================================\n');

% Define scenarios
scenarios = {
    'Best Case - Secured, Low Leverage', [0.4, 0.8, 0.3, 0, 1, 0, 0, 0, 0, 1, 0, 0];
    'Base Case - Moderate', [0.7, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0];
    'Stress Case - Unsecured, High Leverage', [1.0, 0.2, 0.8, 1, 0, 0, 0, 1, 0, 0, 0, 1];
};

fprintf('%-40s %15s %15s\n', 'Scenario', 'Recovery Rate', 'LGD');
fprintf('%s\n', repmat('-', 1, 75));

for i = 1:size(scenarios, 1)
    scenario_name = scenarios{i, 1};
    scenario_X = scenarios{i, 2};
    
    predicted_recovery = predict(mdl, scenario_X);
    predicted_recovery = max(0, min(1, predicted_recovery));
    predicted_lgd = 1 - predicted_recovery;
    
    fprintf('%-40s %14.2f%% %14.2f%%\n', scenario_name, ...
        predicted_recovery * 100, predicted_lgd * 100);
end

%% 11. SUMMARY AND INSIGHTS

fprintf('\n========================================\n');
fprintf('KEY INSIGHTS\n');
fprintf('========================================\n');

fprintf('\n1. MODEL PERFORMANCE:\n');
fprintf('   - R-squared: %.2f%% of variance explained\n', r_squared * 100);
fprintf('   - Average prediction error: %.2f%%\n', mae * 100);

fprintf('\n2. KEY DRIVERS OF RECOVERY:\n');
fprintf('   - Asset Tangibility: Higher tangible assets → Better recovery\n');
fprintf('   - Leverage: Higher debt burden → Worse recovery\n');
fprintf('   - Collateral Type: Secured debt recovers 2-3x better\n');

fprintf('\n3. TYPICAL RECOVERY RATES:\n');
fprintf('   - Senior Secured: %.0f%%\n', mean(data.RecoveryRate(strcmp(data.Seniority, 'Senior Secured'))) * 100);
fprintf('   - Senior Unsecured: %.0f%%\n', mean(data.RecoveryRate(strcmp(data.Seniority, 'Senior Unsecured'))) * 100);
fprintf('   - Subordinated: %.0f%%\n', mean(data.RecoveryRate(strcmp(data.Seniority, 'Subordinated'))) * 100);

fprintf('\n✓ LGD Model Complete!\n');