import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
import streamlit as st
import yfinance as yf

#dự án về vẽ biểu đồ trong giá mở cửa, giá đóng cửa trong python 
#dự án này giúp cho các nhà đầu tư có thể quan sát chung về giá mở cửa
#giá đóng cửa của các mã cổ phiếu từ đấy có thể đưa ra một phần quyết định 
# trong việc đầu tư 
#trong thực tế có thể dùng đoạn code dưới đây để xem tần 
yf.pdr_override()
today = date.today()
d1 = today
end_date = d1
d2 = date.today() - timedelta(days=360)

start_date = d2

st.title("Real-time Stock Price Data")


a = st.text_input("Enter Any Company >>:")  
data = pdr.DataReader(a, start=start_date, end=end_date)
fig, ax = plt.subplots() 
ax = data["Open"].plot(figsize=(12, 8), title=a+" Stock Prices", fontsize=20, label="Open Price")
ax = data["Close"].plot(figsize=(12, 8), title=a+" Stock Prices", fontsize=20, label="Open Price")
plt.legend()
plt.grid()
st.pyplot(fig)
#để chạy file chọn run python file trước sau đó chạy run streamlit 'Tên file'.py dưới terminal