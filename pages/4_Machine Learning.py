# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import inspect
import textwrap
import pandas as pd
import numpy as np
import altair as alt
from utils import show_code
from urllib.error import URLError

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.preprocessing import LabelEncoder


def data_frame_demo():

    # def read_data():
    #     st.markdown("## Upload dataset yang akan diprediksi di sini!")
    #     uploaded_file = st.file_uploader("Choose a file")
    #     dataframe = pd.DataFrame()
    #     if uploaded_file is not None:
    #         # To read file as df:
    #         dataframe = pd.read_csv(uploaded_file)
    #         st.write(dataframe)

    uploaded_file = st.file_uploader("Choose a file")
    df = pd.DataFrame()
    if uploaded_file is not None:
        # To read file as df:
        df = pd.read_csv(uploaded_file)
        df.interpolate(inplace=True)
        df.dropna(inplace=True)
        df.drop('employee_id',axis=1,inplace=True)
        df.drop('region',axis=1,inplace=True)
        
        l=LabelEncoder()
        for i in df.columns:
            if df[i].dtype == 'object':
                df[i]=l.fit_transform(df[i])
        
        mms=MinMaxScaler(feature_range=(0,1))
        df=mms.fit_transform(df)
        df=pd.DataFrame(df)
        st.write(df)
        
   

    if st.button('Prediksi'):
        st.write('Kamu berhasil melakukan prediksi !')
        
        loaded_model = pickle.load(open('model.pickle', "rb"))

        # you can use loaded model to compute predictions
        y_predicted = loaded_model.predict(df)
        st.write(y_predicted)

    else:
        st.write('Hasil prediksi akan muncul disini')
    


st.set_page_config(page_title="Machine learning", page_icon="📊")
st.markdown("# Machine Learning - Prediksi Promosi Pekerja")
st.sidebar.header("Machine Learning - Prediksi Promosi Pekerja")
st.write(
    """Demo ini menunjukan penggunaan streamlit untuk menampilkan data dalam bentuk web app yang mudah dilihat dan di menegrti, kasus machine learningn yang digunakan dalah bagaimana cara untuk memprediksi promosi karyawan"""
)

data_frame_demo()

