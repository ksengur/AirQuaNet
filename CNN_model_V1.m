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
  features = (data(:, 2:12));
  features = (features-min(features(:)))./(max(features(:))-min(features(:)));%7674x12

output = (data(:,14)); %7674x1

% Veriyi eğitim ve test olarak ayır
cv = cvpartition(size(features, 1), 'HoldOut', 0.2);

X_train = features(training(cv), :)';
Y_train = output(training(cv), :);
X_test = features(test(cv), :)';
Y_test = output(test(cv), :);

X_train = num2cell(X_train, 1)';
X_test = num2cell(X_test, 1)';
% Y_train = num2cell(Y_train, 1);
% Y_test = num2cell(Y_test, 1);


number_conv_layer=4;
number_conv_layer2=1;
filtersize=64;

lgraph = layerGraph;

layer_1= sequenceInputLayer(size(X_train{1}, 1),'Name','input', 'Normalization', 'zscore');
lgraph = addLayers(lgraph,layer_1);

for id=1:number_conv_layer
conv1 = convolution1dLayer(3, filtersize, 'Padding', 'same', 'Name', ['conv1_' num2str(id)]);
conv2 = convolution1dLayer(6, filtersize, 'Padding', 'same', 'Name', ['conv2_' num2str(id)]);
conv3 = convolution1dLayer(9, filtersize, 'Padding', 'same', 'Name', ['conv3_' num2str(id)]);
conv4 = convolution1dLayer(1, filtersize, 'Padding', 'same', 'Name', ['conv4_' num2str(id)]);

lgraph = addLayers(lgraph, conv1);
lgraph = addLayers(lgraph, conv2);
lgraph = addLayers(lgraph, conv3);
lgraph = addLayers(lgraph, conv4);

Layer= [concatenationLayer(1, 4, 'Name', ['concat_' num2str(id)])
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5, 'Name', ['dropout' num2str(id)])
        ];
lgraph = addLayers(lgraph, Layer);



if(mod(id,2)==0)

lgraph = addLayers(lgraph,additionLayer(2,"Name","Add_"+id));
lgraph = connectLayers(lgraph, ['dropout' num2str(id)], "Add_"+id+"/in"+1);
lgraph = connectLayers(lgraph, ['concat_' num2str(id-1)], "Add_"+id+"/in"+2);
       
end
      
if(id==2)

   %layer = [flattenLayer('Name', 'flatten') selfAttentionLayer(8,256, 'Name','SA1')  functionLayer(@reflattenSpatialDimensions, 'Name', 'unflatten','Formattable',1) ];
   layer = [flattenLayer('Name', 'flatten') selfAttentionLayer(8,256, 'Name','SA1')  ];

   lgraph = addLayers(lgraph, layer);
   lgraph = connectLayers(lgraph, "Add_"+id, 'flatten');

end

if id==1

lgraph = connectLayers(lgraph, 'input', ['conv1_' num2str(id)]);
lgraph = connectLayers(lgraph, 'input', ['conv2_' num2str(id)]);
lgraph = connectLayers(lgraph, 'input', ['conv3_' num2str(id)]);
lgraph = connectLayers(lgraph, 'input', ['conv4_' num2str(id)]);

else
    if(mod(id,2)==0)
lgraph = connectLayers(lgraph, ['dropout' num2str(id-1)], ['conv1_' num2str(id)]);
lgraph = connectLayers(lgraph, ['dropout' num2str(id-1)], ['conv2_' num2str(id)]);
lgraph = connectLayers(lgraph, ['dropout' num2str(id-1)], ['conv3_' num2str(id)]);
lgraph = connectLayers(lgraph, ['dropout' num2str(id-1)], ['conv4_' num2str(id)]);  
    else
        if(id~=3)
            lgraph = connectLayers(lgraph, ['Add_' num2str(id-1)], ['conv1_' num2str(id)]);
            lgraph = connectLayers(lgraph, ['Add_' num2str(id-1)], ['conv2_' num2str(id)]);
            lgraph = connectLayers(lgraph, ['Add_' num2str(id-1)], ['conv3_' num2str(id)]);
            lgraph = connectLayers(lgraph, ['Add_' num2str(id-1)], ['conv4_' num2str(id)]);  
        else
           
            lgraph = connectLayers(lgraph, 'SA1', ['conv1_' num2str(id)]);
            lgraph = connectLayers(lgraph, 'SA1', ['conv2_' num2str(id)]);
            lgraph = connectLayers(lgraph, 'SA1', ['conv3_' num2str(id)]);
            lgraph = connectLayers(lgraph, 'SA1', ['conv4_' num2str(id)]);  

        end

    end
end


lgraph = connectLayers(lgraph, ['conv1_' num2str(id)] , "concat_"+id+"/in"+1);
lgraph = connectLayers(lgraph, ['conv2_' num2str(id)] , "concat_"+id+"/in"+2);
lgraph = connectLayers(lgraph, ['conv3_' num2str(id)] , "concat_"+id+"/in"+3);
lgraph = connectLayers(lgraph, ['conv4_' num2str(id)] , "concat_"+id+"/in"+4);




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
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 1024, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_test, Y_test}, ...
    'Plots', 'training-progress');
 net = trainNetwork(X_train, Y_train, lgraph, options);
 Y_pred = predict(net, X_test);

% Calculate RMSE
rmse = sqrt(mean((Y_pred - Y_test).^2));

% Calculate R²
SS_res = sum((Y_test - Y_pred).^2);       % Residual sum of squares
SS_tot = sum((Y_test - mean(Y_test)).^2); % Total sum of squares
R2 = 1 - (SS_res / SS_tot);  

% Calculate MAE
mae = mean(abs(Y_pred - Y_test));

% Calculate MAPE
mape = mean(abs((Y_test - Y_pred) ./ Y_test)) * 100;

% Display results
disp(['RMSE: ', num2str(rmse)]);
disp(['R2: ', num2str(R2)]);
disp(['MAE: ', num2str(mae)]);
disp(['MAPE: ', num2str(mape), '%']);


% Plot predictions vs actual
figure;
plot(Y_test, 'b');
hold on;
plot(Y_pred, 'r');
xlabel('Sample Index');
ylabel('Output');
legend('Actual', 'Predicted');
title('Proposed Model: Prediction vs Actual');