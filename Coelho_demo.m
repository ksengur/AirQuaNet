% Load and preprocess data Coelho code
clc
clear
close all
data = (table2array(readtable('.\air_quality_health_impact_data.csv'))); 
 X = data(:, 2:13); % 12 predictor variables (e.g., air quality and weather features)
Y = data(:, 14); 
data = [X Y];
normData = normalize(data(:, 1:end-1)); % Normalize features
targets = data(:, end);
cv = cvpartition(size(data, 1), 'HoldOut', 0.25);
trainData = normData(training(cv), :); trainTargets = targets(training(cv), :);
testData = normData(test(cv), :); testTargets = targets(test(cv), :);

% Feature engineering (PCA example)
[coeff, trainDataPCA, ~, ~, explained] = pca(trainData);
numComponents = find(cumsum(explained) >= 90, 1); % 90% variance
trainDataPCA = trainDataPCA(:, 1:numComponents);
testDataPCA = testData * coeff(:, 1:numComponents);

% Model training with hyperparameter tuning (SVM example)
paramGrid = struct('KernelScale', linspace(0.1, 10, 10), ...
                   'BoxConstraint', logspace(-2, 2, 10));
bestModel = fitrsvm(trainDataPCA, trainTargets, ...
    'KernelFunction', 'gaussian', ...
    'OptimizeHyperparameters', {'KernelScale', 'BoxConstraint'}, ...
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName', 'expected-improvement-plus'));

% Evaluation
predictions = predict(bestModel, testDataPCA);
rmse = sqrt(mean((predictions - testTargets).^2));
r2 = 1 - sum((predictions - testTargets).^2) / sum((mean(testTargets) - testTargets).^2);

% Visualization
scatter(testTargets, predictions, 'filled');
xlabel('Actual Values'); ylabel('Predicted Values');
title(sprintf('Predicted vs Actual (R^2: %.4f, RMSE: %.4f)', r2, rmse));
grid on;
