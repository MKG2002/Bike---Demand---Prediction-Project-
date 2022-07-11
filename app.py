import streamlit as st
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Bike Demand Prediction")

date = st.text_input('Date', 'DD/MM/YYYY')
hour = st.selectbox('Hour of Day (0-23 as per 24 Hour format)', df['Hour'].unique())
hotness = st.number_input('Hotness recorded at a particular hour(in Kelvin)')
humidity = st.number_input('Humidity (in %)')
wind_speed = st.number_input('Wind Speed (in km/hr)')
visibility = st.number_input('Visibility (in m)')
solar_radiation = st.number_input('Recorded in MJ/m^2')
rainfall = st.number_input('Recorded in mm')
snowfall = st.number_input('Recorded in cm')
season = st.selectbox('current season', df['Seasons'].unique())
holiday = st.selectbox('Holiday or not', df['Holiday'].unique())
working_day = st.selectbox('Working day or not', df['Working Day'].unique())

d = pd.DataFrame()
d['id'] = [1]
d['Date'] = [date]
d['Hour'] = [hour]
d['Hotness'] = [hotness]
d['Humidity'] = [humidity]
d['Wind Speed'] = [wind_speed]
d['Visibility'] = [visibility]
d['Solar Radiation'] = [solar_radiation]
d['Rainfall'] = [rainfall]
d['Snowfall'] = [snowfall]
d['Seasons'] = [season]
d['Holiday'] = [holiday]
d['Working Day'] = [working_day]


def func(df):
    df['Hour'] = df['Hour'].map({"0:00":  0,   "1:00":  1, "2:00":  2, "3:00":  3, "4:00":  4, "5:00":  5, "6:00":  6, "7:00":  7, "8:00":  8, "9:00":  9, "10:00": 10, "11:00": 11, "12:00": 12, "13:00": 13, "14:00": 14, "15:00": 15, "16:00": 16, "17:00": 17, "18:00": 18, "19:00": 19, "20:00": 20, "21:00": 21, "22:00": 22, "23:00": 23
                                 })
    return df


def one_hot_encod(dframe, column):
    dframe = pd.concat([dframe, pd.get_dummies(
        dframe[column], prefix=column, drop_first=True)], axis=1)
    dframe = dframe.drop([column], axis=1)
    return dframe


def preprocess(df):
    df2 = df
    df2['Date'] = pd.to_datetime(df2['Date'], format="%d/%m/%Y")
    df2['day'] = df2['Date'].dt.dayofweek
    df2['month'] = df2['Date'].dt.month
    df2 = df2.drop(columns=['Date'], axis=1)
    cols = ['Holiday', 'Working Day' , 'Seasons']
    df2 = df2.drop(['id'], axis=1)
    df2 = func(df2)
    df2['Holiday_YES'] = df2['Holiday'].apply(lambda x :1 if x== "YES" else 0)
    df2['Working Day_YES'] = df2['Working Day'].apply(lambda x :1 if x== "YES" else 0)
    df2['Seasons_Spring'] = df2['Seasons'].apply(lambda x :1 if x== "Spring" else 0)
    df2['Seasons_Summer'] = df2['Seasons'].apply(lambda x :1 if x== "Summer" else 0)
    df2['Seasons_Winter'] = df2['Seasons'].apply(lambda x :1 if x== "Winter" else 0)
    # df2 = df2.drop(['Seasons'], axis=1)
    for col in cols:
        df2 = df2.drop([col] , axis=1)
    return df2



# if st.button('Show Datframe'):
#     st.dataframe(d)

if st.button('Get the Bike Demand Count'):
    d = preprocess(d)
    st.title("The predicted Bike Count According to given conditions is around :- " +
             str(int(np.round(pipe.predict(d)[0]))))
