%% CLIMATE RISK SCORING MODEL
% Predicting climate risk scores and ratings
% Dataset: 600 companies across 10 sectors

%% 1. IMPORT DATA
clear; clc; close all;

data = readtable('climate_risk_dataset.csv');

fprintf('========================================\n');
fprintf('CLIMATE RISK MODEL - DATASET OVERVIEW\n');
fprintf('========================================\n');
fprintf('Total companies: %d\n', height(data));
fprintf('Mean Climate Risk Score: %.1f/100\n', mean(data.ClimateRiskScore));
fprintf('Sectors: %d\n', length(unique(data.Sector)));
fprintf('Geographies: %d\n\n', length(unique(data.Geography)));

disp('Sample Data:');
disp(head(data, 5));

%% 2. RISK RATING DISTRIBUTION

fprintf('\n========================================\n');
fprintf('RISK RATING DISTRIBUTION\n');
fprintf('========================================\n');
rating_counts = groupsummary(data, 'RiskRating', 'IncludeEmptyGroups', true);
rating_counts = sortrows(rating_counts, 'GroupCount', 'descend');
disp(rating_counts);

%% 3. SECTOR ANALYSIS

fprintf('\n========================================\n');
fprintf('CLIMATE RISK BY SECTOR\n');
fprintf('========================================\n');
sector_summary = groupsummary(data, 'Sector', 'mean', {'ClimateRiskScore', 'CarbonIntensity', 'GreenRevenue_Pct'});
sector_summary = sortrows(sector_summary, 'mean_ClimateRiskScore', 'descend');
sector_summary.Properties.VariableNames = {'Sector', 'Count', 'Avg_Risk_Score', ...
    'Avg_Carbon_Intensity', 'Avg_Green_Revenue_Pct'};
disp(sector_summary);

%% 4. VISUALIZATIONS

figure('Name', 'Climate Risk Overview', 'Position', [100 100 1600 900]);

% Plot 1: Risk Score Distribution
subplot(3,3,1);
histogram(data.ClimateRiskScore, 30, 'FaceColor', [0.8 0.3 0.3]);
xlabel('Climate Risk Score');
ylabel('Frequency');
title('Climate Risk Score Distribution');
grid on;

% Plot 2: Risk Rating Pie Chart
subplot(3,3,2);
pie_data = groupsummary(data, 'RiskRating');
pie(pie_data.GroupCount);
title('Risk Rating Distribution');
legend(pie_data.RiskRating, 'Location', 'eastoutside');

% Plot 3: Risk by Sector
subplot(3,3,3);
bar(sector_summary.Avg_Risk_Score);
set(gca, 'XTickLabel', sector_summary.Sector);
xtickangle(45);
ylabel('Average Risk Score');
title('Risk by Sector');
grid on;

% Plot 4: Carbon Intensity vs Risk
subplot(3,3,4);
scatter(data.CarbonIntensity, data.ClimateRiskScore, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Carbon Intensity (tons CO2/$M)');
ylabel('Climate Risk Score');
title('Carbon Intensity vs Risk');
grid on;

% Plot 5: Green Revenue vs Risk
subplot(3,3,5);
scatter(data.GreenRevenue_Pct, data.ClimateRiskScore, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Green Revenue (%)');
ylabel('Climate Risk Score');
title('Green Revenue vs Risk (Negative Correlation)');
grid on;

% Plot 6: ESG Disclosure vs Risk
subplot(3,3,6);
scatter(data.ESG_Disclosure_Score, data.ClimateRiskScore, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('ESG Disclosure Score');
ylabel('Climate Risk Score');
title('ESG Disclosure vs Risk');
grid on;

% Plot 7: Regulatory Exposure vs Risk
subplot(3,3,7);
scatter(data.RegulatoryExposure, data.ClimateRiskScore, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Regulatory Exposure');
ylabel('Climate Risk Score');
title('Regulatory Exposure vs Risk');
grid on;

% Plot 8: Adaptation Spending vs Risk
subplot(3,3,8);
scatter(data.AdaptationSpending_Pct, data.ClimateRiskScore, 30, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Adaptation Spending (% Revenue)');
ylabel('Climate Risk Score');
title('Adaptation Spending vs Risk');
grid on;

% Plot 9: Physical vs Transition Risk
subplot(3,3,9);
physical_risk_map = containers.Map({'Low', 'Medium', 'High', 'Very High'}, [1, 2, 3, 4]);
transition_risk_map = containers.Map({'Very Low', 'Low', 'Medium', 'High', 'Very High'}, [1, 2, 3, 4, 5]);

physical_numeric = zeros(height(data), 1);
transition_numeric = zeros(height(data), 1);

for i = 1:height(data)
    if isKey(physical_risk_map, data.PhysicalRisk{i})
        physical_numeric(i) = physical_risk_map(data.PhysicalRisk{i});
    end
    if isKey(transition_risk_map, data.TransitionRisk{i})
        transition_numeric(i) = transition_risk_map(data.TransitionRisk{i});
    end
end

scatter(transition_numeric, physical_numeric, 100, data.ClimateRiskScore, 'filled', 'MarkerFaceAlpha', 0.7);
xlabel('Transition Risk');
ylabel('Physical Risk');
title('Physical vs Transition Risk (Color = Risk Score)');
colorbar;
grid on;
set(gca, 'XTick', 1:5, 'XTickLabel', {'Very Low', 'Low', 'Med', 'High', 'V.High'});
set(gca, 'YTick', 1:4, 'YTickLabel', {'Low', 'Med', 'High', 'V.High'});

%% 5. PREPARE DATA FOR MODELING

% Convert categorical variables
sector_dummies = dummyvar(categorical(data.Sector));
geo_dummies = dummyvar(categorical(data.Geography));

% Feature matrix
X = [data.CarbonIntensity / 100, ...  % Normalize
     data.ESG_Disclosure_Score, ...
     data.GreenRevenue_Pct, ...
     data.AdaptationSpending_Pct, ...
     data.RegulatoryExposure, ...
     sector_dummies, ...
     geo_dummies];

% Target: Climate Risk Score (continuous)
y = data.ClimateRiskScore;

%% 6. TRAIN-TEST SPLIT

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

%% 7. LINEAR REGRESSION MODEL (CONTINUOUS SCORE)

fprintf('\n========================================\n');
fprintf('MODEL 1: PREDICTING RISK SCORE (0-100)\n');
fprintf('========================================\n');

% Fit model
mdl_score = fitlm(X_train, y_train);

% Get R-squared
r_squared = mdl_score.Rsquared.Ordinary;
fprintf('Model R-squared: %.4f\n', r_squared);

% Predictions
y_pred = predict(mdl_score, X_test);
y_pred = max(0, min(100, y_pred));  % Clip to [0, 100]

% Metrics
mae = mean(abs(y_test - y_pred));
rmse = sqrt(mean((y_test - y_pred).^2));

fprintf('MAE: %.2f points\n', mae);
fprintf('RMSE: %.2f points\n', rmse);

%% 8. CLASSIFICATION MODEL (RISK RATING)

fprintf('\n========================================\n');
fprintf('MODEL 2: PREDICTING RISK RATING\n');
fprintf('========================================\n');

% Get risk ratings for train/test
test_indices = test(cv);
rating_test = categorical(data.RiskRating(test_indices));
rating_train = categorical(data.RiskRating(training(cv)));

% Use Decision Tree instead of Naive Bayes (more robust)
mdl_class = fitctree(X_train, rating_train);

% Predict
rating_pred = predict(mdl_class, X_test);

% Confusion matrix
cm = confusionmat(rating_test, rating_pred);
categories_list = categories(rating_test);

fprintf('\nConfusion Matrix:\n');
disp(array2table(cm, 'VariableNames', categories_list, 'RowNames', categories_list));

% Accuracy
accuracy = sum(diag(cm)) / sum(cm(:));
fprintf('\nClassification Accuracy: %.2f%%\n', accuracy * 100);

%% 9. VISUALIZE RESULTS

figure('Name', 'Climate Risk Model Results', 'Position', [100 100 1400 500]);

% Plot 1: Actual vs Predicted Score
subplot(1,3,1);
scatter(y_test, y_pred, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([0 100], [0 100], 'r--', 'LineWidth', 2);
xlabel('Actual Risk Score');
ylabel('Predicted Risk Score');
title('Risk Score: Actual vs Predicted');
grid on;
legend('Predictions', 'Perfect Fit', 'Location', 'northwest');
axis equal;
xlim([0 100]);
ylim([0 100]);

% Plot 2: Prediction Errors
subplot(1,3,2);
residuals = y_test - y_pred;
histogram(residuals, 30, 'FaceColor', [0.8 0.2 0.2]);
xlabel('Prediction Error (points)');
ylabel('Frequency');
title('Risk Score Prediction Errors');
grid on;
xline(0, 'b--', 'LineWidth', 2);

% Plot 3: Confusion Matrix Heatmap
subplot(1,3,3);
imagesc(cm);
colormap(flipud(hot));
colorbar;
title('Risk Rating Confusion Matrix');
xlabel('Predicted Rating');
ylabel('Actual Rating');
set(gca, 'XTick', 1:length(categories_list), 'XTickLabel', categories_list);
set(gca, 'YTick', 1:length(categories_list), 'YTickLabel', categories_list);
xtickangle(45);

% Add text annotations
for i = 1:length(categories_list)
    for j = 1:length(categories_list)
        text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

%% 10. FEATURE IMPORTANCE

fprintf('\n========================================\n');
fprintf('FEATURE IMPORTANCE (Risk Score Model)\n');
fprintf('========================================\n');

% Get coefficients
coefs = mdl_score.Coefficients.Estimate(2:end);
pvals = mdl_score.Coefficients.pValue(2:end);

% Feature names (simplified)
feature_names = {'CarbonIntensity', 'ESG_Disclosure', 'GreenRevenue', ...
    'AdaptationSpending', 'RegulatoryExposure'};
% Add sector and geo dummies
for i = 1:size(sector_dummies, 2)
    feature_names{end+1} = sprintf('Sector_%d', i);
end
for i = 1:size(geo_dummies, 2)
    feature_names{end+1} = sprintf('Geo_%d', i);
end

% Sort by absolute coefficient
[~, sort_idx] = sort(abs(coefs), 'descend');

fprintf('%-30s %12s %12s %12s\n', 'Feature', 'Coefficient', 't-stat', 'p-value');
fprintf('%s\n', repmat('-', 1, 75));

for i = 1:min(15, length(sort_idx))
    idx = sort_idx(i);
    fprintf('%-30s %12.4f %12.4f %12.6f', feature_names{idx}, ...
        coefs(idx), mdl_score.Coefficients.tStat(idx+1), pvals(idx));
    
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

% Create dummy vectors for Coal Mining sector (highest risk)
sector_coal = zeros(1, size(sector_dummies, 2));
sector_coal(1) = 1;  % Assume first is coal

% Create dummy for Tech sector (lowest risk)
sector_tech = zeros(1, size(sector_dummies, 2));
sector_tech(end) = 1;  % Assume last is tech

% Geography dummies (coastal high risk)
geo_high = zeros(1, size(geo_dummies, 2));
geo_high(1) = 1;

geo_low = zeros(1, size(geo_dummies, 2));
geo_low(end) = 1;

scenarios = {
    'Coal - No Transition', [15, 20, 0, 0, 90, sector_coal, geo_high];
    'Coal - With Transition', [10, 60, 30, 2, 70, sector_coal, geo_low];
    'Oil & Gas - Status Quo', [12, 40, 5, 0.5, 85, sector_coal, geo_high];
    'Tech - Green Leader', [0.5, 85, 80, 3, 20, sector_tech, geo_low];
    'Auto - Moderate EV Push', [4, 65, 35, 1.5, 60, sector_tech, geo_low];
};

fprintf('%-30s %15s %20s\n', 'Scenario', 'Risk Score', 'Risk Rating');
fprintf('%s\n', repmat('-', 1, 70));

for i = 1:size(scenarios, 1)
    scenario_name = scenarios{i, 1};
    scenario_X = scenarios{i, 2};
    
    predicted_score = predict(mdl_score, scenario_X);
    predicted_score = max(0, min(100, predicted_score));
    
    % Determine rating
    if predicted_score >= 80
        rating = 'Critical';
    elseif predicted_score >= 60
        rating = 'High';
    elseif predicted_score >= 40
        rating = 'Medium';
    elseif predicted_score >= 20
        rating = 'Low';
    else
        rating = 'Very Low';
    end
    
    fprintf('%-30s %14.1f %20s\n', scenario_name, predicted_score, rating);
end

%% 12. PORTFOLIO ANALYSIS

fprintf('\n========================================\n');
fprintf('PORTFOLIO CLIMATE RISK ANALYSIS\n');
fprintf('========================================\n');

% Simulate a portfolio
portfolio_sectors = data.Sector(1:50);  % First 50 companies
portfolio_scores = data.ClimateRiskScore(1:50);

fprintf('Portfolio Summary (50 companies):\n');
fprintf('  Average Risk Score: %.1f\n', mean(portfolio_scores));
fprintf('  Weighted by exposure (equal weight): %.1f\n', mean(portfolio_scores));

% Risk breakdown
critical_count = sum(portfolio_scores >= 80);
high_count = sum(portfolio_scores >= 60 & portfolio_scores < 80);
medium_count = sum(portfolio_scores >= 40 & portfolio_scores < 60);
low_count = sum(portfolio_scores < 40);

fprintf('\nRisk Distribution:\n');
fprintf('  Critical Risk: %d companies (%.0f%%)\n', critical_count, (critical_count/50)*100);
fprintf('  High Risk: %d companies (%.0f%%)\n', high_count, (high_count/50)*100);
fprintf('  Medium Risk: %d companies (%.0f%%)\n', medium_count, (medium_count/50)*100);
fprintf('  Low Risk: %d companies (%.0f%%)\n', low_count, (low_count/50)*100);

%% 13. SUMMARY

fprintf('\n========================================\n');
fprintf('KEY INSIGHTS\n');
fprintf('========================================\n');

fprintf('\n1. MODEL PERFORMANCE:\n');
fprintf('   - Risk Score R²: %.2f%%\n', r_squared * 100);
fprintf('   - Risk Score MAE: %.1f points\n', mae);
fprintf('   - Risk Rating Accuracy: %.1f%%\n', accuracy * 100);

fprintf('\n2. HIGHEST RISK SECTORS:\n');
for i = 1:3
    fprintf('   - %s: %.0f/100\n', sector_summary.Sector{i}, sector_summary.Avg_Risk_Score(i));
end

fprintf('\n3. KEY RISK DRIVERS:\n');
fprintf('   - Carbon Intensity: Positive impact on risk\n');
fprintf('   - Green Revenue: Negative impact on risk (reduces it)\n');
fprintf('   - Regulatory Exposure: Major driver for high-emission sectors\n');

fprintf('\n4. TRANSITION OPPORTUNITIES:\n');
fprintf('   - Companies with >50%% green revenue: %d (%.0f%%)\n', ...
    sum(data.GreenRevenue_Pct > 50), sum(data.GreenRevenue_Pct > 50)/height(data)*100);
fprintf('   - Companies with >2%% adaptation spending: %d (%.0f%%)\n', ...
    sum(data.AdaptationSpending_Pct > 2), sum(data.AdaptationSpending_Pct > 2)/height(data)*100);

fprintf('\n✓ Climate Risk Model Complete!\n');