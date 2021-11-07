import streamlit as st
import streamlit.components.v1 as components
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from scipy.stats import norm

import automlbuilder as amb

def data_analysis(workingData, target_att, modelType, analysisPlotsPath, modellingDataPath, log_list, analysisReportsPath):
    dataEdited = False
    log_list = amb.update_logging(log_list, 'Data Analysis', 'Starting Data Analysis')
    st.title('AutoML ' + modelType + ' - Data Analysis')

    st.markdown('## Clean Dataset')
    st.dataframe(workingData.astype('object'))

    with st.form('drop_columns'):
        st.markdown('## Data Configurator')
        st.markdown('#### Select columns to drop from dataset...')
        dataColumns = list(workingData.columns)
        dataColumns.remove(target_att)
        dropSelect = st.multiselect('Select columns...',dataColumns, key='drop_Columns')
        st.markdown('#### Select columns to apply log transformation (numeric only)...')
        st.markdown('Data that does not have a "Normal Distribution" can effect how a machine learning model works. This function applies a log transformation upon the data of the selected attribute(s). You can also use functionality in the model environment setup to apply normalisation to all attributes.')
        normaliseSelect = st.multiselect('Select columns...', dataColumns, key='normalise_Columns')
        st.markdown('#### One-Hot Encode data columns...')
        st.markdown('A new binary attribute will be created for each unique value within the selected attribute.')
        dummySelect = st.multiselect('Select columns...', dataColumns, key='dummy_Columns')
        st.markdown('#### Remove outliers from the dataset')
        st.markdown('An outlier within the dataset is defined as any value that is more than 3 standard deviations from the mean. The following function will remove all rows of data that contain an outlier.')
        outlierSelect = st.radio('Remove Outliers from numeric data',('Yes','No'), index=1)
        st.markdown('##### You must click on "Submit" to make changes')
        submitted = st.form_submit_button("Submit")
        if submitted:
            editedData = workingData.copy()
            if dropSelect != None:
                editedData.drop(dropSelect, axis=1, inplace=True)
                log_list = amb.update_logging(log_list, 'Data Analysis', 'Dropping selected columns - '+str(dropSelect))
            if normaliseSelect != None:
                for col in normaliseSelect:
                    if is_numeric_dtype(editedData[col]):
                        st.markdown('Distribution before log transformation of '+col)
                        st.markdown (col +' skew = '+str(workingData[col].skew()))
                        log_list = amb.update_logging(log_list, 'Data Analysis',
                                                      'Applying log transform to ' + col + ': skew before log transformation =' + str(
                                                          workingData[col].skew()))
                        fig, ax = plt.subplots()
                        ax = sns.distplot(workingData[col], fit=norm)
                        st.pyplot(fig)
                        editedData[col] = np.log1p(editedData[col])
                        plt.clf()
                        st.markdown('Distribution after log transformation of '+col)
                        st.markdown('skew after log transformation =' + str(editedData[col].skew()))
                        fig, ax = plt.subplots()
                        ax = sns.distplot(editedData[col], fit=norm)
                        st.pyplot(fig)
                        log_list = amb.update_logging(log_list, 'Data Analysis',
                                                      'Applying log transform to '+col+': skew after log transformation =' + str(editedData[col].skew()))
            if dummySelect != None:
                for col in dummySelect:
                    #if is_numeric_dtype(editedData[col]) == False:
                    editedData = pd.get_dummies(editedData, columns=[col])
                    log_list = amb.update_logging(log_list, 'Data Analysis', 'One Hot Encoding '+col+' - Dataset Shape after = ' + str(editedData.shape))
            if outlierSelect == 'Yes':
                st.markdown('Dataset Shape before = ' + str(workingData.shape))
                amb.drop_numerical_outliers(editedData)
                st.markdown('Dataset Shape after = ' + str(editedData.shape))
                log_list = amb.update_logging(log_list, 'Data Analysis', 'Removing outliers - Dataset Shape after = ' + str(editedData.shape))
            editedData = amb.clean_data(editedData)
            log_list = amb.update_logging(log_list, 'Data Analysis',
                                          'Cleaning Data - Dataset Shape after = ' + str(editedData.shape))
            st.markdown('## Updated Dataset')
            st.dataframe(editedData.astype('object'))
            editedData.to_csv(modellingDataPath + 'Modelling_Data.csv', index=False, )
            amb.csv_to_html(editedData, '#0096FD', modellingDataPath, 'Modelling_Data.html')
            workingData = editedData
            dataEdited = True
        else:
            workingData.to_csv(modellingDataPath+'Modelling_Data.csv', index=False,)
            amb.csv_to_html(workingData, '#0096FD', modellingDataPath, 'Modelling_Data.html')

    if not os.path.isfile(analysisReportsPath+'OriginalData_Report.html') or dataEdited:
        log_list = amb.update_logging(log_list, 'Data Analysis',
                                      'Generating Sweetviz data reports')
        amb.generate_sv(workingData, target_att, 'OriginalData', None, False,
                                                     analysisReportsPath)
    if not os.path.isfile(analysisReportsPath + 'pp_OriginalData_Report.html') or dataEdited:
        log_list = amb.update_logging(log_list, 'Data Analysis',
                                      'Generating Pandas Profiling data reports')
        amb.generate_pp(workingData, analysisReportsPath)

    st.markdown('## Data Report')
    st.markdown('Two reports are generated using open source libraries, "Pandas Profiling" and "Sweetviz". These reports identify the statistics related to each of the attributes within the datset and the interactions and correlations between atrributes')
    dataReporting = st.selectbox('Select data report:', ('Pandas Profile','Sweetviz'), index=0)
    if dataReporting == 'Sweetviz':
        if os.path.isfile(analysisReportsPath+'OriginalData_Report.html'):
            HtmlFile = open(analysisReportsPath+'OriginalData_Report.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code, height=1000, scrolling=True)
        else:
            st.write('Failed to generate Sweetviz report')
            log_list = amb.update_logging(log_list, 'Data Analysis', 'Failed to generate Sweetviz report')
    else:
        if os.path.isfile(analysisReportsPath + 'pp_OriginalData_Report.html'):
            HtmlFile = open(analysisReportsPath + 'pp_OriginalData_Report.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code, height=1000, scrolling=True)
        else:
            st.write ('Failed to generate Pandas Profile report')
            log_list = amb.update_logging(log_list, 'Data Analysis', 'Failed to generate Pandas Profile report')

    if not os.path.isfile(analysisPlotsPath + 'Box_Plot.png') or dataEdited:
        log_list = amb.update_logging(log_list, 'Data Analysis', 'Generating data visualisation plots')
        amb.half_masked_corr_heatmap(workingData, analysisPlotsPath + 'Correlation.png')
        amb.gen_scatterplots(workingData, target_att, analysisPlotsPath + 'Scatter.png')
        amb.gen_histograms(workingData, analysisPlotsPath + 'Histograms.png')
        amb.gen_boxplots(workingData, analysisPlotsPath + 'Box_Plot.png')
        amb.generate_distPlot(workingData, target_att, analysisPlotsPath)

    st.markdown('## Quick Data Visualisation')
    st.markdown('Quick visualisations generates plots for all attributes within the dataset including distribution plots, box plots, histograms, scatter plots for each attribute against the target and correlation plots')
    filenames = os.listdir(analysisPlotsPath)
    filenames.sort()
    selected_filename = st.selectbox('Select a visualisation:', filenames)

    img = amb.load_image(analysisPlotsPath+selected_filename)
    st.image(img,use_column_width=True)
    plt.close()

    st.markdown('##### Select "' + modelType + ' Model Builder" in App Navigation to continue to Modelling.')

    return log_list, dataEdited

def app():
    modelType = st.session_state['modelType']
    target_att = st.session_state['target_att']
    dataPath = st.session_state['dataPath']
    modellingDataPath = st.session_state['modellingDataPath']
    analysisReportsPath = st.session_state['analysisReportsPath']
    analysisPlotsPath = st.session_state['analysisPlotsPath']
    log_list = st.session_state['log_list']

    if 'data_edited' not in st.session_state:
        st.session_state['data_edited'] = False

    if os.path.isfile(modellingDataPath + 'Modelling_Data.csv'):
        workingData = pd.read_csv(modellingDataPath + 'Modelling_Data.csv')
    else:
        workingData = pd.read_csv(dataPath + 'Clean_Data.csv')

    log_list, dataEdited = data_analysis(workingData, target_att, modelType, analysisPlotsPath, modellingDataPath, log_list, analysisReportsPath)

    st.session_state['log_list'] = log_list
    st.session_state['data_edited'] = dataEdited




