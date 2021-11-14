import streamlit as st
import pandas as pd
import automlbuilder as amb
import base64
import shutil
import os
from pandas.api.types import is_numeric_dtype

# Custom imports
from multipage import MultiPage
from pages import data_analysis, build_classifier, build_regression, explainability, logging # import your pages here

# Create an instance of the app
app = MultiPage()

def initiate_application_cache(modelType, target_att, projectName, dataPath, modellingModelPath, modellingReportsPath, modellingDataPath, analysisReportsPath, analysisPlotsPath, modellingAnalysisPath, explainabilityPath):
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
    if 'modellingAnalysisPath' not in st.session_state:
        st.session_state['modellingAnalysisPath'] = modellingAnalysisPath
    if 'explainabilityPath' not in st.session_state:
        st.session_state['explainabilityPath'] = explainabilityPath

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
    if 'fixImbalance' not in st.session_state:
        st.session_state['fixImbalance'] = False
    if 'featureInteraction' not in st.session_state:
        st.session_state['featureInteraction'] = False
    if 'featureRatio' not in st.session_state:
        st.session_state['featureRatio'] = False
    if 'combineLevels' not in st.session_state:
        st.session_state['combineLevels'] = True


def main():
    if 'log_list' not in st.session_state:
        log_list = []
        st.session_state['log_list'] = log_list

    if 'log_read' not in st.session_state:
        log_read = False
        st.session_state['log_read'] = log_read

    if 'project_created' not in st.session_state:
        projectCreated = False
        st.session_state['project_created'] = projectCreated

    log_read = st.session_state['log_read']
    log_list = st.session_state['log_list']
    projectCreated = st.session_state['project_created']

    # Title of the main page
    st.title("AutoML Model Builder")
    st.markdown('The "AutoML Model Builder" uses an open source machine learning library (PyCaret) to develop machine learning models within this no-code web application.')
    st.markdown('From this application you can load data (csv files only at the moment), perform data analysis, develop either classification or regression machine learning models and then analyse your models performance. All of the data, reports and visualisations created by this application are then saved into a project folder which can then be used for further development.')
    st.markdown('In the sidebar opposite upload your data, give your project a name, select the model type you wish to develop along with the target attribute that your model will use.')
    st.markdown('Once you have completed the setup the CSV file is saved to the project folder and the dataset is then cleaned by removing any attributes that contain all unique values such as ID columns or index columns, drops empty and single valued attributes as well as empty and duplicate rows of data and standardises the attribute names.')
    st.markdown('The cleaned dataset is then saved separately from the original data and will be used for analysis and modelling.')

    st.sidebar.header("Setup")
    selectProject = st.sidebar.selectbox('Project...', ('New','Existing'))
    if selectProject == 'Project...':
        st.stop()
    elif selectProject == 'New':
        uploaded_file = st.sidebar.file_uploader("Choose a csv file", type='csv')
        if uploaded_file is not None:
            uploadData = pd.read_csv(uploaded_file)
            workingData = uploadData.copy()
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
                    if is_numeric_dtype(workingData[target_att]):
                        workingData[target_att] = workingData[target_att].astype('int32')

            message = 'Project Name = '+projectName+' - Model Type = '+modelType+' - Target Attribute = '+target_att+' '
            if not projectCreated:
                log_list = amb.update_logging(log_list, 'Project Setup', message)

            analysisPlotsPath, analysisReportsPath, modellingDataPath, modellingReportsPath, modellingModelPath, dataPath, modellingAnalysisPath, explainabilityPath = amb.generate_working_directories(projectName)
            if not projectCreated:
                log_list = amb.update_logging(log_list, 'Project Setup', 'Project Folder Created')

            uploadData.to_csv(dataPath + 'Original_Data.csv', index=False)
            amb.csv_to_html(uploadData,'#FF7B7B',dataPath, 'Original_Data.html')
            original_shape = uploadData.shape
            if not projectCreated:
                log_list = amb.update_logging(log_list, 'Project Setup', 'Shape of Original Data - '+str(original_shape))

            workingData.to_csv(dataPath + 'Clean_Data.csv', index=False)
            amb.csv_to_html(workingData, '#519000', dataPath, 'Clean_Data.html')
            clean_shape = workingData.shape
            if not projectCreated:
                log_list = amb.update_logging(log_list, 'Project Setup', 'Shape of Cleaned Data - ' + str(clean_shape))

            file = open(dataPath + "Project_Setup.txt", "w")
            file.write(modelType + "\n")
            file.write(target_att + "\n")
            file.close()
            projectCreated = True
            st.session_state['project_created'] = projectCreated
        else:
            st.stop()
    else:
        searchPath ='./Projects'
        directoryContent = os.listdir(searchPath)

        for item in directoryContent:
            if item == '.DS_Store':
                directoryContent.remove(item)
        directoryContent.insert(0,'Select Project...')

        projectName = st.sidebar.selectbox('Select Existing Project:', (directoryContent))
        if projectName != 'Select Project...':
            lines = []
            with open('./Projects/'+projectName+'/data/Project_Setup.txt') as f:
                lines = f.readlines()
                modelType = str(lines[0].strip())
                target_att = str(lines[1].strip())

            analysisPlotsPath, analysisReportsPath, modellingDataPath, modellingReportsPath, modellingModelPath, dataPath, modellingAnalysisPath, explainabilityPath = amb.generate_working_directories(
                projectName)
            if not log_read:
                try:
                    loggingData = pd.read_csv(dataPath + 'Project_log.csv')
                    log_list = loggingData.values.tolist()
                    log_list = amb.update_logging(log_list, 'Project Setup', 'Load Existing Project - ' + projectName)
                    log_read = True
                except:
                    log_list = st.session_state['log_list']
        else:
            st.stop()

    initiate_application_cache(modelType, target_att, projectName, dataPath, modellingModelPath,
                               modellingReportsPath, modellingDataPath, analysisReportsPath, analysisPlotsPath, modellingAnalysisPath, explainabilityPath)
    #print(log_list)
    st.session_state['log_list'] = log_list
    st.session_state['log_read'] = log_read

    # Add all your applications (pages) here
    app.add_page("Data Analysis", data_analysis.app)
    if modelType == 'Classification':
        app.add_page("Classifier Model Builder", build_classifier.app)
    else:
        app.add_page("Regression Model Builder", build_regression.app)
    app.add_page("Model Explainability", explainability.app)
    app.add_page("Project Log", logging.app)
    # The main app
    app.run()

if __name__ == "__main__":
    main()
