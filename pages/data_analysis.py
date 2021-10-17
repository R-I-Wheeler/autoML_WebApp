import streamlit as st
import streamlit.components.v1 as components
import os
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np


import automlbuilder as amb

def data_analysis(workingData, target_att, modelType, sv_report, svReportSuccess, pp_report, ppReportSuccess, plotsPath, modellingDataPath):
    st.title('AutoML ' + modelType + ' - Data Analysis')

    st.markdown('## Original Dataset')
    st.dataframe(workingData.astype('object'))

    st.markdown('## Data Report')
    dataReporting = st.selectbox('Select data report:', ('Pandas Profile','Sweetviz'), index=0)
    if dataReporting == 'Sweetviz':
        if svReportSuccess == True:
            components.html(sv_report, width=1000,height=1000, scrolling=True)
        else:
            st.write('Failed to generate Sweetviz report')
    else:
        if ppReportSuccess == True:
            components.html(pp_report, width=1000, height=1000, scrolling=True)
        else:
            st.write ('Failed to generate Pandas Profile report')

    st.markdown('## Quick Data Visualisation')

    filenames = os.listdir(plotsPath)
    selected_filename = st.selectbox('Select a visualisation:', filenames)

    img = amb.load_image(plotsPath+selected_filename)
    st.image(img,use_column_width=True)
    plt.close()

    transformTarget = False
    with st.form('drop_columns'):
        st.markdown('## Data Configurator')
        st.markdown('#### Select columns to drop from dataset...')
        dataColumns = list(workingData.columns)
        dataColumns.remove(target_att)
        dropSelect = st.multiselect('Select columns...',dataColumns, key='drop_Columns')
        dataColumns = list(workingData.columns)
        st.markdown('#### Select columns to apply log transformation (numeric only)...')
        normaliseSelect = st.multiselect('Select columns...', dataColumns, key='normalise_Columns')
        st.markdown('#### Encode categorical data columns...')
        dummySelect = st.multiselect('Select columns...', dataColumns, key='dummy_Columns')
        outlierSelect = st.radio('Remove Outliers from numeric data',('Yes','No'), index=1)
        submitted = st.form_submit_button("Submit")
        if submitted:
            editedData = workingData.copy()
            if dropSelect != None:
                editedData.drop(dropSelect, axis=1, inplace=True)
            if normaliseSelect != None:
                for col in normaliseSelect:
                    if is_numeric_dtype(editedData[col]):
                        if col == target_att:
                            transformTarget = True
                        else:
                            editedData[col] = np.log1p(workingData[col])
                st.pyplot(plt = editedData.hist(bins=50, figsize=(15, 15)))

            if dummySelect != None:
                for col in dummySelect:
                    if is_numeric_dtype(editedData[col]) == False:
                        editedData = pd.get_dummies(editedData, columns=[col])
            if outlierSelect == 'Yes':
                amb.drop_numerical_outliers(editedData)
            st.markdown('## Dataset')
            st.dataframe(editedData.astype('object'))
            editedData.to_csv(modellingDataPath + 'Modelling_Data.csv', index=False, )
        else:
            workingData.to_csv(modellingDataPath+'Modelling_Data.csv', index=False,)
    return (workingData, transformTarget)

def app():
    modelType = st.session_state['modelType']
    target_att = st.session_state['target_att']
    dataPath = st.session_state['dataPath']
    modellingDataPath = st.session_state['modellingDataPath']
    analysisReportsPath = st.session_state['analysisReportsPath']
    analysisPlotsPath = st.session_state['analysisPlotsPath']

    workingData = pd.read_csv(dataPath + 'Original_Data.csv')

    amb.half_masked_corr_heatmap(workingData, analysisPlotsPath + 'correlation.png')
    amb.gen_scatterplots(workingData, target_att, analysisPlotsPath + 'scatter.png')
    amb.gen_histograms(workingData, analysisPlotsPath + 'histogram.png')
    amb.gen_boxplots(workingData, analysisPlotsPath + 'boxplot.png')

    sv_report, svReportSuccess = amb.generate_sv(workingData, target_att, 'OriginalData', None, False,
                                                 analysisReportsPath)
    pp_report, ppReportSuccess = amb.generate_pp(workingData, analysisReportsPath)

    workingdata, transformTarget = data_analysis(workingData, target_att, modelType, sv_report, svReportSuccess,
                                                 pp_report, ppReportSuccess, analysisPlotsPath, modellingDataPath)

    st.session_state['workingdata'] = workingdata
    if 'transformTarget' not in st.session_state:
        st.session_state['transformTarget'] = transformTarget


