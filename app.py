import streamlit as st
import pandas as pd
import automlbuilder as amb

# Custom imports
from multipage import MultiPage
from pages import data_analysis, build_classifier, build_regression, explainability # import your pages here

# Create an instance of the app
app = MultiPage()

def initiate_application_cache(modelType, target_att, projectName, dataPath, modellingModelPath, modellingReportsPath, modellingDataPath, analysisReportsPath, analysisPlotsPath):
    if 'modelType' not in st.session_state:
        st.session_state['modelType'] = modelType
    if 'target_att' not in st.session_state:
        st.session_state['target_att'] = target_att
    if 'projectName' not in st.session_state:
        st.session_state['projectName'] = projectName
    if 'dataPath' not in st.session_state:
        st.session_state['dataPath'] = dataPath
    if 'modellingModelPath' not in st.session_state:
        st.session_state['modellingModelPath'] = modellingModelPath
    if 'modellingReportsPath' not in st.session_state:
        st.session_state['modellingReportsPath'] = modellingReportsPath
    if 'modellingDataPath' not in st.session_state:
        st.session_state['modellingDataPath'] = modellingDataPath
    if 'analysisReportsPath' not in st.session_state:
        st.session_state['analysisReportsPath'] = analysisReportsPath
    if 'analysisPlotsPath' not in st.session_state:
        st.session_state['analysisPlotsPath'] = analysisPlotsPath
    if 'transformTarget' not in st.session_state:
        st.session_state['transformTarget'] = False
    if 'dataEdited' not in st.session_state:
        st.session_state['dataEdited'] = False

    if 'activateNormalise' not in st.session_state:
        st.session_state['activateNormalise'] = False
    if 'normaliseMethod' not in st.session_state:
        st.session_state['normaliseMethod'] = 'zscore'
    if 'activateTransform' not in st.session_state:
        st.session_state['activateTransform'] = False
    if 'transformMethod' not in st.session_state:
        st.session_state['transformMethod'] = 'yeo-johnson'
    if 'targetTransform' not in st.session_state:
        st.session_state['targetTransform'] = False
    if 'targetMethod' not in st.session_state:
        st.session_state['targetMethod'] = 'box-cox'
    if 'combineLevels' not in st.session_state:
        st.session_state['combineLevels'] = True

def main():
    # Title of the main page
    st.title("AutoML Model Builder")

    st.sidebar.header("Project Setup")
    uploaded_file = st.sidebar.file_uploader("Choose a csv file", type='csv')
    if uploaded_file is not None:
        workingData = pd.read_csv(uploaded_file)
        projectName = st.sidebar.text_input('Name Project...', 'myProject')
        modelType = st.sidebar.selectbox('Select Model Type:', ('Classification', 'Regression'))
        workingData = amb.clean_data(workingData)
        column_list = list(workingData.columns)
        column_list.insert(0,'Select Target Attribute...')
        target_att = st.sidebar.selectbox('Select Target Attribute:', (column_list))

        if target_att == 'Select Target Attribute...':
            st.stop()
        else:
            if modelType == 'Classification':
                workingData[target_att] = workingData[target_att].astype('int32')
        analysisPlotsPath, analysisReportsPath, modellingDataPath, modellingReportsPath, modellingModelPath, dataPath = amb.generate_working_directories(projectName)

        workingData.to_csv(dataPath + 'Original_Data.csv', index=False)

        initiate_application_cache(modelType, target_att, projectName, dataPath, modellingModelPath,
                                   modellingReportsPath, modellingDataPath, analysisReportsPath, analysisPlotsPath)

        # Add all your applications (pages) here
        app.add_page("Data Analysis", data_analysis.app)
        if modelType == 'Classification':
            app.add_page("Classifier Model Builder", build_classifier.app)
        else:
            app.add_page("Regression Model Builder", build_regression.app)
        app.add_page("Model Explainability", explainability.app)
        # The main app
        app.run()
    else:
        st.stop()

if __name__ == "__main__":
    main()
