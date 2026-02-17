%% PROBABILITY OF DEFAULT (PD) MODELING
% Corporate Credit Risk - Logistic Regression
% Dataset: 2000 firms with financial ratios

%% 1. IMPORT AND EXPLORE DATA
clear; clc; close all;

% Import dataset
data = readtable('corporate_credit_pd_dataset.csv');

fprintf('========================================\n');
fprintf('DATASET OVERVIEW\n');
fprintf('========================================\n');
fprintf('Total Firms: %d\n', height(data));
fprintf('Total Defaults: %d\n', sum(data.Default));
fprintf('Default Rate: %.2f%%\n\n', mean(data.Default) * 100);

% Display first few rows
disp('Sample Data (First 10 rows):');
disp(data(1:10, :));

%% 2. EXPLORATORY DATA ANALYSIS

fprintf('\n========================================\n');
fprintf('SUMMARY STATISTICS\n');
fprintf('========================================\n');

% Numeric variables
numeric_vars = {'LeverageRatio', 'InterestCoverage', 'ProfitMargin', ...
                'CurrentRatio', 'EquityVolatility'};

disp(array2table(data{:, numeric_vars}, 'VariableNames', numeric_vars));
disp(' ');
disp('Descriptive Statistics:');
disp(data(:, numeric_vars));
summary(data(:, numeric_vars))

% Sector distribution
fprintf('\n========================================\n');
fprintf('SECTOR DISTRIBUTION\n');
fprintf('========================================\n');
sector_summary = groupsummary(data, 'Sector', {'sum', 'mean'}, 'Default');
sector_summary.Properties.VariableNames{3} = 'Total_Defaults';
sector_summary.Properties.VariableNames{4} = 'Default_Rate';
sector_summary.Default_Rate = sector_summary.Default_Rate * 100;
disp(sector_summary);

%% 3. COMPARE DEFAULTED VS NON-DEFAULTED

fprintf('\n========================================\n');
fprintf('DEFAULTED VS NON-DEFAULTED COMPARISON\n');
fprintf('========================================\n');

defaulted = data(data.Default == 1, :);
non_defaulted = data(data.Default == 0, :);

comparison = table();
comparison.Metric = numeric_vars';
comparison.Non_Defaulted = [
    mean(non_defaulted.LeverageRatio);
    mean(non_defaulted.InterestCoverage);
    mean(non_defaulted.ProfitMargin);
    mean(non_defaulted.CurrentRatio);
    mean(non_defaulted.EquityVolatility)
];
comparison.Defaulted = [
    mean(defaulted.LeverageRatio);
    mean(defaulted.InterestCoverage);
    mean(defaulted.ProfitMargin);
    mean(defaulted.CurrentRatio);
    mean(defaulted.EquityVolatility)
];
comparison.Difference = comparison.Defaulted - comparison.Non_Defaulted;

disp(comparison);

%% 4. VISUALIZATIONS - UNIVARIATE ANALYSIS

figure('Name', 'Univariate Analysis', 'Position', [100 100 1400 900]);

subplot(3,2,1);
histogram(non_defaulted.LeverageRatio, 30, 'FaceAlpha', 0.5, 'FaceColor', 'b');
hold on;
histogram(defaulted.LeverageRatio, 30, 'FaceAlpha', 0.5, 'FaceColor', 'r');
xlabel('Leverage Ratio');
ylabel('Frequency');
title('Leverage Ratio Distribution');
legend('Non-Defaulted', 'Defaulted');
grid on;

subplot(3,2,2);
histogram(non_defaulted.InterestCoverage, 30, 'FaceAlpha', 0.5, 'FaceColor', 'b');
hold on;
histogram(defaulted.InterestCoverage, 30, 'FaceAlpha', 0.5, 'FaceColor', 'r');
xlabel('Interest Coverage');
ylabel('Frequency');
title('Interest Coverage Distribution');
legend('Non-Defaulted', 'Defaulted');
grid on;

subplot(3,2,3);
histogram(non_defaulted.ProfitMargin, 30, 'FaceAlpha', 0.5, 'FaceColor', 'b');
hold on;
histogram(defaulted.ProfitMargin, 30, 'FaceAlpha', 0.5, 'FaceColor', 'r');
xlabel('Profit Margin');
ylabel('Frequency');
title('Profit Margin Distribution');
legend('Non-Defaulted', 'Defaulted');
grid on;

subplot(3,2,4);
histogram(non_defaulted.CurrentRatio, 30, 'FaceAlpha', 0.5, 'FaceColor', 'b');
hold on;
histogram(defaulted.CurrentRatio, 30, 'FaceAlpha', 0.5, 'FaceColor', 'r');
xlabel('Current Ratio');
ylabel('Frequency');
title('Current Ratio Distribution');
legend('Non-Defaulted', 'Defaulted');
grid on;

subplot(3,2,5);
histogram(non_defaulted.EquityVolatility, 30, 'FaceAlpha', 0.5, 'FaceColor', 'b');
hold on;
histogram(defaulted.EquityVolatility, 30, 'FaceAlpha', 0.5, 'FaceColor', 'r');
xlabel('Equity Volatility');
ylabel('Frequency');
title('Equity Volatility Distribution');
legend('Non-Defaulted', 'Defaulted');
grid on;

subplot(3,2,6);
bar(sector_summary.Default_Rate);
set(gca, 'XTickLabel', sector_summary.Sector);
ylabel('Default Rate (%)');
title('Default Rate by Sector');
grid on;

%% 5. PREPARE DATA FOR MODELING

fprintf('\n========================================\n');
fprintf('PREPARING DATA FOR MODELING\n');
fprintf('========================================\n');

% Create dummy variables for sector (one-hot encoding)
sector_dummies = dummyvar(categorical(data.Sector));
sector_names = categories(categorical(data.Sector));

% Combine features
X = [data.LeverageRatio, data.InterestCoverage, data.ProfitMargin, ...
     data.CurrentRatio, data.EquityVolatility, sector_dummies];

y = data.Default;

% Feature names for interpretation
feature_names = [numeric_vars, sector_names'];

fprintf('Features: %d\n', size(X, 2));
fprintf('Samples: %d\n', size(X, 1));
fprintf('Positive class (defaults): %d (%.2f%%)\n', sum(y), mean(y)*100);

%% 6. TRAIN-TEST SPLIT

fprintf('\n========================================\n');
fprintf('TRAIN-TEST SPLIT\n');
fprintf('========================================\n');

% 70-30 split
rng(42);  % For reproducibility
cv = cvpartition(length(y), 'HoldOut', 0.30);

X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

fprintf('Training set: %d samples (%.2f%% defaults)\n', ...
    length(y_train), mean(y_train)*100);
fprintf('Test set: %d samples (%.2f%% defaults)\n', ...
    length(y_test), mean(y_test)*100);

%% 7. LOGISTIC REGRESSION MODEL

fprintf('\n========================================\n');
fprintf('TRAINING LOGISTIC REGRESSION\n');
fprintf('========================================\n');

% Fit generalized linear model (binomial family = logistic regression)
mdl = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');

% Display model summary
disp(mdl);

% Extract and display coefficients
coefficients = mdl.Coefficients;
fprintf('\nModel Coefficients:\n');
fprintf('%-25s %10s %10s %10s %10s\n', 'Variable', 'Estimate', 'SE', 't-stat', 'p-value');
fprintf('%-25s %10s %10s %10s %10s\n', repmat('-', 1, 25), repmat('-', 1, 10), ...
    repmat('-', 1, 10), repmat('-', 1, 10), repmat('-', 1, 10));

for i = 1:length(feature_names)
    var_name = feature_names{i};
    if i <= length(numeric_vars)
        display_name = var_name;
    else
        display_name = ['Sector_' var_name];
    end
    
    coef = coefficients.Estimate(i+1);  % +1 to skip intercept
    se = coefficients.SE(i+1);
    tstat = coefficients.tStat(i+1);
    pval = coefficients.pValue(i+1);
    
    fprintf('%-25s %10.4f %10.4f %10.4f %10.6f', ...
        display_name, coef, se, tstat, pval);
    
    if pval < 0.001
        fprintf(' ***\n');
    elseif pval < 0.01
        fprintf(' **\n');
    elseif pval < 0.05
        fprintf(' *\n');
    else
        fprintf('\n');
    end
end

fprintf('\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n');

%% 8. PREDICTIONS ON TEST SET

fprintf('\n========================================\n');
fprintf('MODEL EVALUATION - TEST SET\n');
fprintf('========================================\n');

% Predict probabilities
y_pred_prob = predict(mdl, X_test);

% Predict classes using 0.5 threshold
y_pred = y_pred_prob >= 0.5;
y_test = double(y_test);
y_pred = double(y_pred);

% Confusion matrix
fprintf('\nConfusion Matrix:\n');
cm = confusionmat(y_test, y_pred);
disp(array2table(cm, 'VariableNames', {'Pred_0', 'Pred_1'}, ...
    'RowNames', {'Actual_0', 'Actual_1'}));

% Calculate metrics
TN = cm(1,1);
FP = cm(1,2);
FN = cm(2,1);
TP = cm(2,2);

accuracy = (TP + TN) / sum(cm(:));
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);
specificity = TN / (TN + FP);

fprintf('\nPerformance Metrics:\n');
fprintf('  Accuracy:    %.2f%%\n', accuracy * 100);
fprintf('  Precision:   %.2f%%\n', precision * 100);
fprintf('  Recall:      %.2f%%\n', recall * 100);
fprintf('  Specificity: %.2f%%\n', specificity * 100);
fprintf('  F1-Score:    %.2f%%\n', f1_score * 100);

%% 9. ROC CURVE AND AUC

figure('Name', 'ROC Curve', 'Position', [100 100 800 600]);

% Calculate ROC curve
[X_roc, Y_roc, thresholds, AUC] = perfcurve(y_test, y_pred_prob, 1);

% Plot ROC curve
plot(X_roc, Y_roc, 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.3f)', AUC));
legend('Model', 'Random Classifier', 'Location', 'southeast');
grid on;

fprintf('\n========================================\n');
fprintf('ROC-AUC Score: %.3f\n', AUC);
fprintf('========================================\n');

% Interpretation
if AUC >= 0.9
    fprintf('Interpretation: Excellent discrimination\n');
elseif AUC >= 0.8
    fprintf('Interpretation: Good discrimination\n');
elseif AUC >= 0.7
    fprintf('Interpretation: Acceptable discrimination\n');
else
    fprintf('Interpretation: Poor discrimination\n');
end

%% 10. CALIBRATION PLOT

figure('Name', 'Calibration Plot', 'Position', [100 100 800 600]);

% Bin predictions
n_bins = 10;
bin_edges = linspace(0, 1, n_bins + 1);
bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
observed_freq = zeros(1, n_bins);
predicted_freq = zeros(1, n_bins);

for i = 1:n_bins
    in_bin = y_pred_prob >= bin_edges(i) & y_pred_prob < bin_edges(i+1);
    if i == n_bins
        in_bin = y_pred_prob >= bin_edges(i) & y_pred_prob <= bin_edges(i+1);
    end
    
    if sum(in_bin) > 0
        observed_freq(i) = mean(y_test(in_bin));
        predicted_freq(i) = mean(y_pred_prob(in_bin));
    end
end

% Plot calibration
plot(predicted_freq, observed_freq, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
xlabel('Predicted Probability');
ylabel('Observed Frequency');
title('Calibration Plot');
legend('Model', 'Perfect Calibration', 'Location', 'northwest');
grid on;
axis([0 1 0 1]);

%% 11. THRESHOLD OPTIMIZATION

fprintf('\n========================================\n');
fprintf('THRESHOLD OPTIMIZATION\n');
fprintf('========================================\n');

% Try different thresholds
thresholds_to_try = 0.05:0.05:0.50;
results = table();

for thresh = thresholds_to_try
    y_pred_thresh = y_pred_prob >= thresh;
    cm_thresh = confusionmat(double(y_test), double(y_pred_thresh));
    
    TP = cm_thresh(2,2);
    TN = cm_thresh(1,1);
    FP = cm_thresh(1,2);
    FN = cm_thresh(2,1);
    
    acc = (TP + TN) / sum(cm_thresh(:));
    prec = TP / (TP + FP);
    rec = TP / (TP + FN);
    f1 = 2 * (prec * rec) / (prec + rec);
    
    results = [results; table(thresh, acc, prec, rec, f1)];
end

results.Properties.VariableNames = {'Threshold', 'Accuracy', 'Precision', ...
    'Recall', 'F1_Score'};

disp('Performance at Different Thresholds:');
disp(results);

% Find optimal threshold (maximize F1)
[~, optimal_idx] = max(results.F1_Score);
optimal_threshold = results.Threshold(optimal_idx);

fprintf('\nOptimal Threshold: %.2f (maximizes F1-score)\n', optimal_threshold);
fprintf('F1-Score at optimal threshold: %.2f%%\n', ...
    results.F1_Score(optimal_idx) * 100);

%% 12. FEATURE IMPORTANCE

fprintf('\n========================================\n');
fprintf('FEATURE IMPORTANCE\n');
fprintf('========================================\n');

% Based on standardized coefficients (z-scores)
% Higher absolute value = more important

feature_importance = abs(coefficients.Estimate(2:end));  % Exclude intercept
[sorted_importance, sorted_idx] = sort(feature_importance, 'descend');

fprintf('Top 10 Most Important Features:\n');
fprintf('%-30s %15s\n', 'Feature', 'Abs(Coefficient)');
fprintf('%-30s %15s\n', repmat('-', 1, 30), repmat('-', 1, 15));

for i = 1:min(10, length(sorted_idx))
    idx = sorted_idx(i);
    if idx <= length(numeric_vars)
        feat_name = numeric_vars{idx};
    else
        feat_name = ['Sector_' sector_names{idx - length(numeric_vars)}];
    end
    fprintf('%-30s %15.4f\n', feat_name, sorted_importance(i));
end

%% 13. SUMMARY AND RECOMMENDATIONS

fprintf('\n========================================\n');
fprintf('ANALYSIS SUMMARY\n');
fprintf('========================================\n');

fprintf('\nModel Performance:\n');
fprintf('  AUC-ROC: %.3f\n', AUC);
fprintf('  Accuracy: %.2f%%\n', accuracy * 100);
fprintf('  Precision: %.2f%%\n', precision * 100);
fprintf('  Recall: %.2f%%\n', recall * 100);

fprintf('\nKey Risk Factors (Strongest Predictors):\n');
for i = 1:3
    idx = sorted_idx(i);
    coef = coefficients.Estimate(idx + 1);
    if idx <= length(numeric_vars)
        feat_name = numeric_vars{idx};
    else
        feat_name = ['Sector: ' sector_names{idx - length(numeric_vars)}];
    end
    
    if coef > 0
        direction = 'INCREASES';
    else
        direction = 'DECREASES';
    end
    
    fprintf('  %d. %s %s default probability\n', i, feat_name, direction);
end

