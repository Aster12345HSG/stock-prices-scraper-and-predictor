# Author: Michal Piotr Kaczynski, No. 19-617-810, 
# CodingXCamp nickname: Aster12345

# The following is a simple piece of code that scrapes stock market financial data from Yahoo Finance
# After scraping, it saves basic statistical data for desired stocks and time periods as csv files on the computer
# After saving them, it imports those files back to Python for analysis and future price prediction

# Acknowledgments to: R Junaid Raza for source code that served as base for scraping stock data
# Acknowledgments to: Data Flair for providing me with basic knowledge about predicion models and to ChatGPT for helping me build one
# Acknowledgments to: ChatGPT that helped me debug the code

# In order to run this code on your computer, you should have:
# 1. Internet connection
# 2. Installed the following libraries: pandas_datareader, datetime, yfinance, pandas, prophet, matplotlib (install using pip install in command prompt)

# I run this code on my computer in Visual Studio Code on Python 3.11.3 64-bit from Microsoft Store and it works with this configuration

# Import the necessary libraries and packages
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()  # Override the pandas_datareader with yfinance
import pandas as pd
import os

# Tickers list
# Tickers are short symbols for companies' names on the stock exchange
# For example, 'TSLA' is a ticker for Tesla, 'GOOGL' is a ticker for 
# Alphabet Inc. that owns Google, and so on...

# I decided to analyse Nvidia, Google and Microsoft, but you can change it
# in the code below and analyse any company you want
ticker_list = ['NVDA','GOOGL','MSFT']

# The user of this code can get data for desired period of time by changing
# the start date and end date below
# Please note that dates follow YYYY-MM-DD format 
start_date = "2015-01-01"
end_date = "2020-01-01"
files = []



def getData(ticker):
    print(ticker)
    # Get data using Yahoo Finance API
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    dataname = ticker + '_' + str(end_date)
    files.append(dataname)
    SaveData(data, dataname)

# The following piece of code creates a folder named 'data' to enable
# saving csv files there. Therefore, the user does not have to create
# any file or folder themselves. It will be created automatically on desktop
# to enhance user experience. 
def SaveData(df, filename):
    if not os.path.exists('./data'):
        os.makedirs('./data')
    # Save DataFrame as CSV file in the './data' folder
    df.to_csv('./data/' + filename + '.csv')

# This loop will iterate over the ticker list, get data for each ticker, and save it as a file.
for tik in ticker_list:
    getData(tik)

# Read the saved files
for i in range(len(ticker_list)): #'len' ensures that Python will iterate 
                                  #for as many times as there are companies 
                                  # that you selected in ticker_list 
    # Read CSV file from the './data' folder
    df1 = pd.read_csv('./data/' + str(files[i]) + '.csv')

# The code above saves the desired data from Yahoo Finance on my computer 
# in a folder named 'data' created automatically on the desktop with basic
# statistics about the stock.

# Below we start to import those saved files to Python and analyse them
 
# List the filenames of the saved CSV files
files = [f for f in os.listdir('./data') if f.endswith('.csv')]

# Read the saved files and store them in a dictionary
data = {}
for filename in files:
    ticker = filename.split('_')[0]  # Extract the ticker from the filename
    filepath = os.path.join('./data', filename)  # Build the file path
    df = pd.read_csv(filepath)  # Read the CSV file
    data[ticker] = df  # Store the DataFrame in the dictionary

# Importing libraries needed for analysis and price prediction
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to predict future stock prices using Prophet model
def predict_stock_price(df, ticker):
    # Prepare the DataFrame with required columns for Prophet
    data = df[['Date', 'Close']]
    data.columns = ['ds', 'y']

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(data)

    # Number of future time steps to predict
    future_steps = 30  # This is the number of days in the future that the 
                       # the program will predict
                       # Adjust the number of steps as needed

    # Generate future dates for prediction
    future_dates = model.make_future_dataframe(periods=future_steps)

    # Make predictions for future dates
    forecast = model.predict(future_dates)

    # Plot the actual and predicted stock prices
    model.plot(forecast, xlabel='Date', ylabel='Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.show()

# Example usage
for ticker in ticker_list:
    df = data[ticker]
    predict_stock_price(df, ticker)

# Thanks to this program we get graphical analysis of Nvidia, Google and
# Microsoft stock prices and their predicted future value. I have run this 
# program several times for different periods and stocks with past data to
# check retrospectively if it was correct and it seems to have a decent 
# correctness rate of about 70% for predicting them up to a month into the 
# future (which is not as good as 100%, but better than the odds of pure 
# chance which are 50%).

# This program could also be easily adjusted to serve one's needs. I, for 
# example, could use this program to make graphs for my Bachelor Thesis if I
# wrote it about specific stocks and specific periods. In this case, I could 
# scrape Yahoo Finance for specific data and then replace the Prophet based 
# stock prediction model with simple Matplotlib library to create elegant 
# graphs that contain only the data that I need in order to improve my 
# written assignment.  

