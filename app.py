import streamlit as st
import time
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import plotly.express as px
from sklearn import model_selection
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import pandas as pd
import numpy as np
import scipy as sp
# RNN model for forecasting stock prices with Keras and Tensorflow given a company name


def load_data():
    url = "https://github.com/calebsiow0228/STOCK-RNN/blob/main/SP500.csv.zip?raw=true"
    df = pd.read_csv(url, compression = 'zip')
    return df


def stock_price_prediction(company_name, df, epochs=100, batch_size=32, neurons=50, dropout=0.2, loss="mean_squared_error", optimizer='adam', validation_split=0.2, patience=10, verbose=1):
    # select the company from the dataframe
    df = df[df["COMPANY"] == company_name]
    # select the columns that we need
    df = df[["Date", "Close"]]
    # sort the dataframe by date
    df = df.sort_values(by="Date")
    # reset the index
    df = df.reset_index(drop=True)
    # set the date column as index
    df = df.set_index("Date")
    # create a new dataframe with only the close column
    data = df.filter(["Close"])
    # convert the dataframe to a numpy array
    dataset = data.values
    # get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .8))
    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # create the training dataset
    # create the scaled training dataset
    train_data = scaled_data[0:training_data_len, :]
    # split the data into x_train and y_train datasets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()
    # convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # build the LSTM model
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(25))
    model.add(Dense(1))
    # compile the model
    # optimizer SGD
    model.compile(optimizer=optimizer, loss=loss)
    # train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
              callbacks=[EarlyStopping(monitor="val_loss", patience=patience, verbose=verbose, mode="min")])
    # create the testing dataset
    # create a new array containing scaled values from index 1543 to 2003
    test_data = scaled_data[training_data_len - 60:, :]
    # create the datasets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    # convert the data to a numpy array
    x_test = np.array(x_test)
    # reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    # get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid["Predictions"] = predictions
    #plot the data in plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train["Close"],
                    mode='lines',
                    name='Train'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid["Close"],
                    mode='lines',
                    name='Valid'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid["Predictions"],
                    mode='lines',
                    name='Predictions'))
    fig.update_layout(title=company_name, xaxis_title="Date", yaxis_title="Close Price USD ($)")

    return rmse, predictions, model, train, valid, fig

# streamlit app for stock price prediction with RNN model user selects the company name and the model will predict the stock price for the next 30 days


def main():
    st.title("Stock Price Prediction with RNN")
    st.sidebar.title("Stock Price Prediction with RNN")
    st.markdown("This app predicts the stock price of a company using RNN")
    st.sidebar.markdown(
        "This app predicts the stock price of a company using RNN")

    @st.cache(persist=True)
    def load_data():
        # Path to data: /stock/data/SP500.csv
        url = "https://github.com/calebsiow0228/STOCK-RNN/blob/main/SP500.csv.zip?raw=true"
        df = pd.read_csv(
            url, compression='zip')
        return df
    df = load_data()
    # get the unique company names
    company_names = df.COMPANY.unique()
    # sort the company names
    company_names = sorted(company_names)
    # sidebar - company selection
    # sidebar - company selection
    selected_company = st.sidebar.selectbox(
        "Select company for prediction", company_names)
    # number of days for prediction
    days = st.sidebar.slider("Days of prediction:", 1, 30)
    # number of epochs
    epochs = st.sidebar.slider("Number of epochs:", 100, 500)
    # batch size
    batch_size = st.sidebar.slider("Batch size:", 32, 64)
    # neurons
    neurons = st.sidebar.slider(
        "Number of neurons in the 2 LSTM layers:", 30, 100)
    # dropout
    dropout = st.sidebar.slider("Dropout:", 0.1, 0.5)
    # loss function
    loss = st.sidebar.selectbox("Loss function:", [
                                "mean_squared_error", "mean_absolute_error"])
    # optimizer
    optimizer = st.sidebar.selectbox("Optimizer:", [
                                     "adam", "sgd"])
    # validation split
    validation_split = st.sidebar.slider("Validation split:", 0.1, 0.3)
    # patience
    patience = st.sidebar.slider("Patience:", 5, 15)
    # verbose
    verbose = st.sidebar.selectbox("Verbose:", [0, 1])
    # button
    if st.sidebar.button("Predict"):

        # add a progress bar to streamlit while the model is training
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        rmse, predictions, model, train, valid, fig = stock_price_prediction(
            selected_company, df, epochs, batch_size, neurons, dropout, loss, optimizer, validation_split, patience, verbose)
        # save the model
        # create a folder for the model
        if not os.path.exists("models"):
            os.mkdir("models")
        # save the model as a h5 file with the name of the company
        model.save("models/{}.h5".format(selected_company))
        # print the rmse
        st.subheader("RMSE: " + str(rmse))
        # plot the data in plotly
        st.subheader("The Stock Price Predictions For " + selected_company)
        st.plotly_chart(fig)
        # print the predicted price
        
        st.subheader("Predicted price for the next " + str(days) + " days:")
        st.write(predictions[-days:])
        # plot the data
        st.subheader("Predicted data vs Actual data for " + selected_company)
        st.line_chart(valid[["Close", "Predictions"]])
        # plot the data
        st.subheader("Training data vs Actual data for " + selected_company)
        st.line_chart(train[["Close"]])
        # plot the data
        st.subheader("Actual data for " + selected_company)
        st.line_chart(valid[["Close"]])
        # plot the data
        st.subheader("Predicted data for " + selected_company)
        st.line_chart(valid[["Predictions"]])
        # plot the model
        st.subheader("Model summary")
        st.text(model.summary())


if __name__ == "__main__":
    main()
    
#edits to be made: Recall past models to improve speed of results
