# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:59:30 2020

@author: Olusegun
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor


@st.cache(suppress_st_warning=True)
def load_data():
    #url = 'https://data.mendeley.com/datasets/pdnjz7zz5x/1/files/e7007dff-4061-494f-a9f5-16c20f457d1d/100_Batches_IndPenSim.zip?dl=1'
    #print("Downloading file......")
    #raw_file = requests.get(url)
    #print("Download complete!")
    #with open('pen_batches.zip', 'wb') as f:
        #f.write(raw_file.content)
    
    #print('Un-zipping file.......')
    #zip_ref = zipfile.ZipFile('pen_batches.zip', 'r')
    #zip_ref.extractall()
    #print('Un-zipping complete!')
    st.text("Loading data")
    pen_data = pd.read_csv('data/pen_data.csv')
    
    st.text("Data loaded")
    return pen_data


@st.cache
def rename_columns(col_list):
    new_columns = []
    cols_dictionary = {}
    
    
    for col in col_list:
        new_col = col.replace(' ', '_')
        new_col = new_col.replace('/', '_')
        new_col = new_col.split('(')[0]
        new_columns.append(new_col.lower())
        cols_dictionary[new_col] = col
    return new_columns, cols_dictionary
    


@st.cache
def get_offline_data(df):
    pen_data_offline = df.dropna()
    
    pen_data_offline.reset_index(drop=True, inplace=True)
    return pen_data_offline



@st.cache
def split_data(df):
    #X = pen_data_clean.drop(columns=['penicillin_concentration', 'batch_id', 'fault_flag'])
    features = [
                'time_',
                'vessel_volume',
                'carbon_evolution_rate',
                'oil_flow',
                'dissolved_oxygen_concentration',
                'substrate_concentration',
               ]
    X = df[features]
    y = df.offline_penicillin_concentration
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test
@st.cache
def train_model(X_train, y_train):
    
    
    
    gs_gb = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                              learning_rate=0.01, loss='ls', max_depth=4,
                              max_features=1.0, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=1000,
                              n_iter_no_change=None, presort='auto',
                              random_state=42, subsample=0.4, tol=0.0001,
                              validation_fraction=0.1, verbose=0, warm_start=False)
    gs_gb.fit(X_train, y_train)
    return gs_gb

@st.cache
def test_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return rsme
    

st.title("Real-Time Bioprocess Product Monitoring")

page = st.sidebar.selectbox("Choose a page:",
                            ["Homepage",
                             "Penicillin Monitoring",
                             "Online Biomass Monitoring",
                             "Rapid mAb monitoring",
                             "Plasmid quantification monitoring"])

if page == "Homepage":
    st.subheader("Background")
    
    st.markdown("""
                One of the challenges of bioprocessing is product monitoring.
                Bioprocess development and discovery requires optimization of processes 
                for increased product yield. However, due to the analytics 
                available for product measurement, bioprocesss development is 
                slowed. For example, during mAb production using mammalian cell 
                fermentation, monitoring the product yield requires taking samples 
                of broth at interval, spinning down the cells and running analytical 
                processes such as SDS-PAGE and ELISA. For accurate product 
                quantification, HPLC is sometimes combined with the methods 
                mentioned. This can take as much as 24 hours to carry out. This means that development process is halted until the result of product quantification is achieved.
    
    Rapid product quantification is necessary to fast-track product development by 
    removing the product quantification bottleneck associated with current methods. 
    AI/ML products are revolutionizing all industries and the bioprocess industries 
    cannot be left out. By employing machine learning, rapid product monitoring 
    tools can be developed. Taking advantage of the vast amount of sensor data 
    generated during a bioprocess operation will enable the development these 
    tools. For example, a simple microbial batch fermentation generates real time
     data including dissolved oxygen concentration, pH, temperature, time of 
     fermentation, carbon evolution rate, acid flow rate and base flow rate. 
     It is possible to take advantage of these data and develop machine learning 
     models that are able to learn on the data and predict product concentration 
     in real time.
    
                """)
                
    st.subheader("Completed Projects")
    st.markdown("""
                Currently, the following projects have been completed
                
                * Rapid quantification of penicillin concentration
                """)
    st.subheader("Future Projects")
    st.markdown("""
                This project was only executed on penicillin data. Future projects will include
                
                * Rapid quantifiation of biomass concentration
                
                * Rapid quantification of protein yeld (specifically mAbs)
                
                * Real-time quantification of plasmid concentration
                """)

elif page == "Penicillin Monitoring":
    st.subheader("Rapid quantification of penicillin concentration")
    
    st.markdown("""
                 This project demonstrate the possibility of developing a rapid product 
     quantification and monitoring using machine learning. The project a 
     predictive tool for quantifying penicillin concentration during upstream 
     processing. The data used contains 100 batches of penicillin fermentation 
     run at different conditions. There are over 30 features in the dataset that 
     are possible predictors for the penicillin concentration, however based on 
     statistical analysis and domain knowledge, only six features have been 
     selected as predictors.
    The significance of selecting the most important features means, the model 
    can be trained quickly and rapidly deployed.
    
    Please try it out below.
                
                """)
    data = load_data()
    
    new_cols, col_dict = rename_columns(data.columns)
    data.columns = new_cols
    if st.checkbox("Show all data"):
        st.dataframe(data)
    data = get_offline_data(data)
    
    if st.checkbox("Show training data"):
        st.dataframe(data)
    
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    rmse = test_performance(model, X_test, y_test)
    
    st.header("""
              Fermentation parameters:
              """)
    
    rand = np.random.randint(X_test.shape[0]- 1, size=1)[0]
    time = st.number_input("Time (h)", 0., 1000., float(X_test.iloc[rand, 0]))
    vessel = st.number_input("Vessel Volume", 0., 100000., float(X_test.iloc[rand, 1]))
    cer = st.number_input("Carbon Evolution Rate (g/h)", 0., 1000., float(X_test.iloc[rand, 2]))
    of = st.number_input("Oil Flow (L/h)", 0., 1000., float(X_test.iloc[rand, 3]))
    do = st.number_input("Dissolved Oxygen Concentration (mg/L)", 0., 1000., float(X_test.iloc[rand, 4])) 
    sub = st.number_input("Substrate Concentration (g/L)", 0., 1000., float(X_test.iloc[rand, 5])) 
    
    status = st.empty()
    button = st.button('Check Penicillin Concentration')
    #status.text("Please click button abbove to check penicillin Concentration")
    
    if button:
        status.text("Generating Penicillin Concentration")
        input_values = np.array((time, vessel, cer, of, do, sub)).reshape(1, -1)
        prediction = model.predict(input_values)
        status.text("Penicillin  concentration generated")
        st.markdown(f"## **{np.round(prediction, 2)[0]}** g/L")

elif page == "Online Biomass Monitoring":
    st.subheader("Coming soon!")
elif page == "Rapid mAb monitoring":
    st.subheader("Coming soon!")
elif page == "Plasmid quantification monitoring":
    st.subheader("Coming soon!")



