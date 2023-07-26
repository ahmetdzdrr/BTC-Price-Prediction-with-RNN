# BTC-Price-Prediction-with-RNN
Bitcoin price prediction analysis using LSTM models.


RNN (Recurrent Neural Network):
RNN is a type of artificial neural network designed to handle sequential data and time dependencies. Unlike traditional feedforward neural networks, RNNs have a memory feature, allowing them to retain information about previous data points and consider their temporal relationships. This makes them suitable for tasks like natural language processing and time series analysis. However, standard RNNs often struggle with capturing long-term dependencies effectively.

![1_xcSfOV1EQet0EeFY1m1jPA](https://github.com/ahmetdzdrr/BTC-Price-Prediction-with-RNN/assets/117534684/95a8dabe-bda2-4020-aac9-cfed14bdd7d6)

LSTM (Long Short-Term Memory):
LSTM is a specialized type of RNN designed to address the issue of handling long-term dependencies. It introduces memory cells and three types of gates: input, output, and forget gates. These gates allow the model to control the flow of information, retaining essential information from past inputs while forgetting less relevant details. As a result, LSTM networks can effectively capture long-term dependencies in sequential data and are widely used in various applications, including language modeling, machine translation, and financial time series prediction.

![1_7cMfenu76BZCzdKWCfBABA](https://github.com/ahmetdzdrr/BTC-Price-Prediction-with-RNN/assets/117534684/6fab9963-4ce8-4aaa-96ae-cda9d575951c)


Data on Bitcoin prices has been fetched using the API provided by Coingecko for the years 2016 to 2023, including the closing prices. The closing prices for a 60-day period have been used to predict the closing price. Training and testing data have been prepared for the model. After training and testing the data, predictions for the closing price for the next 30 days are sought.

NOTE: This deep learning prediction analysis is by no means an investment recommendation.

![btc-33](https://github.com/ahmetdzdrr/BTC-Price-Prediction-with-RNN/assets/117534684/7cd1a7f6-7213-4242-aab9-a27dd0317709)


LSTM MODEL ARCHITECTURE

The correct LSTM model architecture is as follows:

The model starts with a Sequential container for stacking layers.

The first layer is an LSTM layer with 128 units (cells), and it is set to return sequences (return_sequences=True) because the subsequent LSTM layer requires sequences as input. The input shape is specified as (x_train.shape[1], 1), where x_train.shape[1] represents the number of time steps, and 1 is the number of features at each time step.

A Dropout layer with a dropout rate of 0.2 is added after the first LSTM layer. The Dropout layer helps prevent overfitting by randomly setting 20% of the LSTM layer's output to zero during training.

The second layer is another LSTM layer with 64 units, also set to return sequences.

Another Dropout layer with a dropout rate of 0.2 follows the second LSTM layer.

The third layer is an LSTM layer with 32 units, and it does not return sequences as it is the final LSTM layer in the sequence.

Finally, a Dense layer with 1 unit is added. This layer will be responsible for predicting the output, which in this case is the closing price.

PREDICTION ANALYSIS VISUALIZATION

![Screenshot_12](https://github.com/ahmetdzdrr/BTC-Price-Prediction-with-RNN/assets/117534684/ae2b674b-14ec-4cb2-954b-f12c2b24059a)

FUTURE PREDICTION ANALYSIS VISUALIZATION

![Screenshot_11](https://github.com/ahmetdzdrr/BTC-Price-Prediction-with-RNN/assets/117534684/76faaa6a-b26f-46f8-887c-fb9290080b78)



