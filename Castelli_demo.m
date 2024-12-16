% Load and preprocess data
clc
clear
close all
data = (table2array(readtable('.\air_quality_health_impact_data.csv'))); 
 X = data(:, 2:13); % 12 predictor variables (e.g., air quality and weather features)
Y = data(:, 14); 
data = [X Y];
features = normalize(data(:, 1:end-1)); % Normalize features
targets = data(:, end);
cv = cvpartition(size(data, 1), 'HoldOut', 0.25);
trainFeatures = features(training(cv), :); trainTargets = targets(training(cv), :);
testFeatures = features(test(cv), :); testTargets = targets(test(cv), :);
% data = readtable('data.csv'); % Replace with dataset file
% features = normalize(data{:, 1:end-1}); % Normalize features
% targets = data{:, end};
% cv = cvpartition(size(data, 1), 'HoldOut', 0.3);
% trainFeatures = features(training(cv), :); trainTargets = targets(training(cv));
% testFeatures = features(test(cv), :); testTargets = targets(test(cv));

% PCA for dimensionality reduction
[coeff, trainPCA, ~, ~, explained] = pca(trainFeatures);
numComponents = find(cumsum(explained) >= 90, 1); % 90% variance
trainPCA = trainPCA(:, 1:numComponents); testPCA = testFeatures * coeff(:, 1:numComponents);

% Train SVR with hyperparameter tuning
bestModel = fitrsvm(trainPCA, trainTargets, 'KernelFunction', 'gaussian', ...
    'OptimizeHyperparameters', {'KernelScale', 'BoxConstraint'}, ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus'));

% Evaluate model
predictions = predict(bestModel, testPCA);
rmse = sqrt(mean((predictions - testTargets).^2));
r2 = 1 - sum((predictions - testTargets).^2) / sum((mean(testTargets) - testTargets).^2);
fprintf('RMSE: %.4f, R²: %.4f\n', rmse, r2);

% Visualization
scatter(testTargets, predictions, 'filled');
xlabel('Actual Values'); ylabel('Predicted Values');
title(sprintf('Predicted vs Actual (R²: %.4f, RMSE: %.4f)', r2, rmse)); grid on;
