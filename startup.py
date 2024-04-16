import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('startUp (1).csv')

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-size: 60px; font-family: Helvetica'>STARTUP PROFIT PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Gomycode Data Science Daintree</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.image('pngwing.com.png', width = 600, caption = 'Start Up Project')

st.header('Project Background Information',divider = True)
st.write("The overarching objective of this ambitious project is to meticulously engineer a highly sophisticated predictive model meticulously designed to meticulously assess the intricacies of startup profitability. By harnessing the unparalleled power and precision of cutting-edge machine learning methodologies, our ultimate aim is to furnish stakeholders with an unparalleled depth of insights meticulously delving into the myriad factors intricately interwoven with a startup's financial success. Through the comprehensive analysis of extensive and multifaceted datasets, our mission is to equip decision-makers with a comprehensive understanding of the multifarious dynamics shaping the trajectory of burgeoning enterprises. Our unwavering commitment lies in empowering stakeholders with the indispensable tools and knowledge requisite for making meticulously informed decisions amidst the ever-evolving landscape of entrepreneurial endeavors.")

#Sidebar Designs
st.sidebar.image('pngwing.com 111(1).png', width = 200)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

#User Inputs

rd_spend = st.sidebar.number_input('Research and Development Expense', data['R&D Spend'].min(), data['R&D Spend'].max())

admin = st.sidebar.number_input('Administrative Expense', data['Administration'].min(), data['Administration'].max())

mkt= st.sidebar.number_input('Marketing Spend', data['Marketing Spend'].min(), data['Marketing Spend'].max())

state = st.sidebar.selectbox('Company Location', data['State'].unique())

#import Transformers

admin_scaler = joblib.load('Administration_scaler.pkl')
mkt_scaler = joblib.load('Marketing Spend_scaler.pkl')
rd_spend_scaler = joblib.load('R&D Spend_scaler.pkl')
state_encoder = joblib.load('State_encoder.pkl')

#user input dataframe
user_input = pd.DataFrame()
user_input['R&D Spend'] = [rd_spend]
user_input['Administration'] = [admin]
user_input['Marketing Spend'] = [mkt]
user_input['State'] = [state]


st.markdown("<br>", unsafe_allow_html=True)
st.header('Input Variable', divider = True)
st.dataframe(user_input, use_container_width = True)


# transform users input according to training scale and encoding
user_input['R&D Spend'] = rd_spend_scaler.transform(user_input[['R&D Spend']])
user_input['Administration'] = admin_scaler.transform(user_input[['Administration']])
user_input['Marketing Spend'] = mkt_scaler.transform(user_input[['Marketing Spend']])
user_input['State'] = state_encoder.transform(user_input[['State']])

st.header('Transformed Input Variable', divider = True)
st.dataframe(user_input, use_container_width = True)
