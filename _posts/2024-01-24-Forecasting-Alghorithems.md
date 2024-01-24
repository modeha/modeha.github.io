---
layout: post
title:  "Recurrent Neural Network Models"
date:   2024-01-24 9:31:29 +0900
categories: Update
---
### Recurrent Neural Network Models For Forecasting

LSTM achieves this by learning the weights for internal gates that control the recurrent connections within each node. Although developed for sequence data, LSTMs have not proven effective on time series forecasting problems where the output is a function of recent observations, e.g. an autoregressive type forecasting problem, such as the car sales dataset.

In this section, we will explore three variations on the LSTM model for univariate time series forecasting:
 - Vanilla LSTM: The LSTM network as-is. 
 - CNN-LSTM: A CNN network that learns input features and an LSTM that interprets them.
- ConvLSTM: A combination of CNNs and LSTMs where the LSTM units read input data using the convolutional process of a CNN.