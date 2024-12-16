clc
clear
warning off
close all
data = (table2array(readtable('.\air_quality_health_impact_data.csv'))); 
% columnIndex = 13;
% % Remove rows where the value in the specified column is -200
% data(data(:,columnIndex) == -200, :) = [];

% windowSize = 3; % You can adjust the window size
% features = (replaceMinus200WithMedian(data(:, 1:end-1), windowSize));

% Select specific columns (features and output)
  features = (data(:, 2:13));
  % features = (features-min(features(:)))./(max(features(:))-min(features(:)));%7674x12

output = (data(:,14)); %7674x1

% Veriyi eğitim ve test olarak ayır
cv = cvpartition(size(features, 1), 'HoldOut', 0.25);

X_train = features(training(cv), :)';
Y_train = output(training(cv), :);
X_test = features(test(cv), :)';
Y_test = output(test(cv), :);

X_train = num2cell(X_train, 1)';
X_test = num2cell(X_test, 1)';
% Y_train = num2cell(Y_train, 1);
% Y_test = num2cell(Y_test, 1);


%%

number_conv_layer=1;
% number_conv_layer2=1;
filtersize=32;

lgraph = layerGraph;

layer_1= sequenceInputLayer(size(X_train{1}, 1),'Name','input', 'Normalization', 'zscore');
lgraph = addLayers(lgraph,layer_1);

for id=1:number_conv_layer
    for depth=1:3
conv1 = convolution1dLayer(3, filtersize, 'Padding', 'same', 'Name', ['conv1_' num2str(depth) num2str(id)]);
conv2 = convolution1dLayer(6, filtersize, 'Padding', 'same', 'Name', ['conv2_' num2str(depth) num2str(id)]);
conv3 = convolution1dLayer(9, filtersize, 'Padding', 'same', 'Name', ['conv3_' num2str(depth) num2str(id)]);
conv4 = convolution1dLayer(1, filtersize, 'Padding', 'same', 'Name', ['conv4_' num2str(depth) num2str(id)]);

lgraph = addLayers(lgraph, conv1);
lgraph = addLayers(lgraph, conv2);
lgraph = addLayers(lgraph, conv3);
lgraph = addLayers(lgraph, conv4);


Layer= [concatenationLayer(1, 4, 'Name', ['concat_' num2str(depth) num2str(id)])
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5, 'Name', ['dropout' num2str(depth) num2str(id)])
        ];
lgraph = addLayers(lgraph, Layer);

lgraph = connectLayers(lgraph, ['conv1_' num2str(depth) num2str(id)] , "concat_"+depth+id+"/in"+1);
lgraph = connectLayers(lgraph, ['conv2_' num2str(depth) num2str(id)] , "concat_"+depth+id+"/in"+2);
lgraph = connectLayers(lgraph, ['conv3_' num2str(depth) num2str(id)] , "concat_"+depth+id+"/in"+3);
lgraph = connectLayers(lgraph, ['conv4_' num2str(depth) num2str(id)] , "concat_"+depth+id+"/in"+4);



if (depth~=1)
lgraph = connectLayers(lgraph, ['dropout' num2str(depth-1) num2str(id)], ['conv1_' num2str(depth) num2str(id)]);
lgraph = connectLayers(lgraph, ['dropout' num2str(depth-1) num2str(id)], ['conv2_' num2str(depth) num2str(id)]);
lgraph = connectLayers(lgraph, ['dropout' num2str(depth-1) num2str(id)], ['conv3_' num2str(depth) num2str(id)]);
lgraph = connectLayers(lgraph, ['dropout' num2str(depth-1) num2str(id)], ['conv4_' num2str(depth) num2str(id)]);

end

    end

     
    
    if (id~=number_conv_layer)
    lgraph = addLayers(lgraph,[additionLayer(2,"Name","Add_"+id) flattenLayer('Name', "flatten"+id) selfAttentionLayer(8,256, 'Name',"SA"+id)]);
    lgraph = connectLayers(lgraph, ['dropout' num2str(depth) num2str(id)], "Add_"+id+"/in"+1);
    lgraph = connectLayers(lgraph, ['concat_' num2str(1) num2str(id)], "Add_"+id+"/in"+2);
    else 
    lgraph = addLayers(lgraph,[additionLayer(2,"Name","Add_"+id)]);
    lgraph = connectLayers(lgraph, ['dropout' num2str(depth) num2str(id)], "Add_"+id+"/in"+1);
    lgraph = connectLayers(lgraph, ['concat_' num2str(1) num2str(id)], "Add_"+id+"/in"+2);
    end
   



    if id==1
        lgraph = connectLayers(lgraph, 'input', ['conv1_' num2str(1) num2str(id)]);
        lgraph = connectLayers(lgraph, 'input', ['conv2_' num2str(1) num2str(id)]);
        lgraph = connectLayers(lgraph, 'input', ['conv3_' num2str(1) num2str(id)]);
        lgraph = connectLayers(lgraph, 'input', ['conv4_' num2str(1) num2str(id)]);

      

    else
            lgraph = connectLayers(lgraph, ['SA' num2str(id-1)], ['conv1_' num2str(1) num2str(id)]);
            lgraph = connectLayers(lgraph, ['SA' num2str(id-1)], ['conv2_' num2str(1) num2str(id)]);
            lgraph = connectLayers(lgraph, ['SA' num2str(id-1)], ['conv3_' num2str(1) num2str(id)]);
            lgraph = connectLayers(lgraph, ['SA' num2str(id-1)], ['conv4_' num2str(1) num2str(id)]);  


    end

end

 head_layer=[ 
              globalAveragePooling1dLayer(Name="gap") 
              batchNormalizationLayer
              flattenLayer
              dropoutLayer(0.1)
              fullyConnectedLayer(1)
              regressionLayer];
              lgraph = addLayers(lgraph,head_layer);
              lgraph = connectLayers(lgraph,"Add_"+id,"gap");

analyzeNetwork(lgraph)

%%
options = trainingOptions('adam', ...
    'MaxEpochs', 2000, ...
    'MiniBatchSize', 1024, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_test, Y_test}, ...
    'Plots', 'training-progress');
 net = trainNetwork(X_train, Y_train, lgraph, options);
 Y_pred = predict(net, X_test);

% Calculate RMSE
rmse = sqrt(mean((Y_pred - Y_test).^2));
%Calculation of R²
SS_res = sum((Y_test - Y_pred).^2);       % Residual sum of squares
SS_tot = sum((Y_test - mean(Y_test)).^2); % Total sum of squares
R2 = 1 - (SS_res / SS_tot);  
mae = (mean(abs(Y_pred - Y_test)));
mape = (mean(abs((Y_pred - Y_test)./Y_test)));
% Display RMSE
disp(['RMSE: ', num2str(rmse)]);
disp(['MAE: ', num2str(mae)]);
disp(['MAPE: ', num2str(mape)]);
disp(['R2: ', num2str(R2)]);
% Plot predictions vs actual
figure;
plot(Y_test, 'b');
hold on;
plot(Y_pred, 'r');
xlabel('Sample Index');
ylabel('Output');
legend('Actual', 'Predicted');

% plot(features(:,1))
% xlabel('Number of Samples')
% ylabel ('AQI')
% ylim([-100 600])
% grid on