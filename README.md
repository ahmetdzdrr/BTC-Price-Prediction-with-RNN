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

![download](https://github.com/ahmetdzdrr/BTC-Price-Prediction-with-RNN/assets/117534684/7ed0a2da-cb8d-4755-83e4-a933f923e237)



