import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
from pycaret.classification import load_model

import automlbuilder as amb

def generate_explainability_charts(evaluationData, projectName, modellingModelPath, explainabilityPath):

    instancePath = explainabilityPath+'Instance/'
    amb.create_directory(instancePath+'Force/')
    amb.create_directory(instancePath + 'Waterfall/')

    saved_model = load_model(modellingModelPath+projectName+'_finalised_model')
    train_pipe = saved_model[:-1].transform(evaluationData)

    numRows = train_pipe[train_pipe.columns[0]].count()
    numRows.item()

    if numRows <= 50:
        maximumRows = numRows
    else:
        maximumRows = 50
    maximumRows = getattr(maximumRows, "tolist", lambda: maximumRows)()

    try:
        explainer = shap.TreeExplainer(saved_model.named_steps["trained_model"])
        shap_values = explainer.shap_values(X=train_pipe.iloc[0:maximumRows, :])

        try:
            for data_idx in range(maximumRows):
                forcePlot = shap.force_plot(explainer.expected_value[1], shap_values[1][data_idx, :], train_pipe.iloc[data_idx, :], show=False)
                shap.save_html('forcePlot_instance_'+str(data_idx)+'.html', forcePlot)
        except Exception as e:
            print(e)
            try:
                for data_idx in range(maximumRows):
                    forcePlot = shap.force_plot(explainer.expected_value, shap_values[data_idx, :], train_pipe.iloc[data_idx, :], show=False)
                    shap.save_html('forcePlot_instance_'+str(data_idx)+'.html', forcePlot)
            except Exception as e:
                print(e)
                print('Failed to create SHAP Force Plot')
        sourcePath = './'
        sourceFiles = os.listdir(sourcePath)
        destinationPath = instancePath+'Force/'
        for file in sourceFiles:
            if file.endswith('.html'):
                shutil.move(os.path.join(sourcePath, file), os.path.join(destinationPath, file))
        plt.close()
        try:
            explainAll = shap.force_plot(explainer.expected_value[1], shap_values[1], train_pipe)
            shap.save_html('explainer.html', explainAll)
        except Exception as e:
            print(e)
            try:
                explainAll = shap.force_plot(explainer.expected_value, shap_values, train_pipe)
                shap.save_html('explainer.html', explainAll)
            except Exception as e:
                print(e)
                print('Failed to create SHAP Overall Force Plot')
        plt.close()
        shap.summary_plot(shap_values, train_pipe, show=False)
        plt.savefig('Summary_Plot.png',format = "png",dpi = 150,bbox_inches = 'tight')
        plt.close()
        sourcePath = './'
        sourceFiles = os.listdir(sourcePath)
        destinationPath = explainabilityPath
        for file in sourceFiles:
            if file.endswith('.png') or file.endswith('.html'):
                shutil.move(os.path.join(sourcePath, file), os.path.join(destinationPath, file))
    except Exception as e:
        print(e)
        print('Unable to create TreeExplainer SHAP explainability files')
        try:
            explainer = shap.Explainer(saved_model.named_steps["trained_model"], train_pipe)
            shap_values = explainer(train_pipe)

            for data_idx in range(maximumRows):
                shap.plots.waterfall(shap_values[data_idx], show=False)
                plt.savefig('Waterfall_Plot_Instance_'+str(data_idx)+'.png',format = "png",dpi = 150,bbox_inches = 'tight')
                plt.close()
            sourcePath = './'
            sourceFiles = os.listdir(sourcePath)
            destinationPath = instancePath+'Waterfall/'
            for file in sourceFiles:
                if file.endswith('.png'):
                    shutil.move(os.path.join(sourcePath, file), os.path.join(destinationPath, file))
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.savefig('Summary_Plot.png',format = "png",dpi = 150,bbox_inches = 'tight')
            plt.close()

            shap.plots.heatmap(shap_values, show=False)
            plt.savefig('Heatmap.png',format = "png",dpi = 150,bbox_inches = 'tight')
            plt.close()
            sourcePath = './'
            sourceFiles = os.listdir(sourcePath)
            destinationPath = explainabilityPath
            for file in sourceFiles:
                if file.endswith('.png'):
                    shutil.move(os.path.join(sourcePath, file), os.path.join(destinationPath, file))
        except Exception as e:
            print(e)
            print('Unable to create Explainer files')
            try:
                ex = shap.KernelExplainer(saved_model.named_steps["trained_model"].predict, train_pipe)
                shap_values = ex.shap_values(train_pipe.iloc[0, :])

                for data_idx in range(maximumRows):
                    forcePlot = shap.force_plot(ex.expected_value, shap_values, train_pipe.iloc[data_idx, :])
                    shap.save_html('forcePlot_instance_' + str(data_idx) + '.html', forcePlot)
                sourcePath = './'
                sourceFiles = os.listdir(sourcePath)
                destinationPath = instancePath+'Force/'
                for file in sourceFiles:
                    if file.endswith('.html'):
                        shutil.move(os.path.join(sourcePath, file), os.path.join(destinationPath, file))
                shap_values = ex.shap_values(X=train_pipe)
                shap.summary_plot(shap_values, train_pipe, show=False)
                plt.savefig('Summary_Plot.png',format = "png",dpi = 150,bbox_inches = 'tight')
                plt.close()
                shap.summary_plot(shap_values, train_pipe, plot_type='bar', show=False)
                plt.savefig('Summary_Bar_Plot.png',format = "png",dpi = 150,bbox_inches = 'tight')
                plt.close()
                sourcePath = './'
                sourceFiles = os.listdir(sourcePath)
                destinationPath = explainabilityPath
                for file in sourceFiles:
                    if file.endswith('.png'):
                        shutil.move(os.path.join(sourcePath, file), os.path.join(destinationPath, file))
            except Exception as e:
                print(e)
                print('Unable to generate SHAP explainability files')
    sourcePath = './'
    sourceFiles = os.listdir(sourcePath)
    destinationPath = explainabilityPath
    for file in sourceFiles:
        if file.endswith('.png') or file.endswith('.html'):
            shutil.move(os.path.join(sourcePath, file), os.path.join(destinationPath, file))
    return instancePath

def model_explainability(instancePath, explainabilityPath):

    st.markdown('The folowing shows how each of the features contributes to push the model output from the base value, the average model output over the entire training dataset used to develop the model, to the predicted output of the selected instance of the unseen data.')
    st.markdown('Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.')

    filenames = os.listdir(instancePath+'Force/')
    filenames.sort()
    if filenames:
        selected_filename = st.selectbox('Select a visualisation:', filenames, index=0)
        HtmlFile = open(instancePath+'Force/'+selected_filename, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        #print(source_code)
        components.html(source_code, height=200)
    else:
        filenames = os.listdir(instancePath + 'Waterfall/')
        filenames.sort()
        selected_filename = st.selectbox('Select a visualisation:', filenames, index=0)
        img = amb.load_image(instancePath + 'Waterfall/' + selected_filename)
        st.image(img, use_column_width=True)
        plt.close()

    my_file = Path(explainabilityPath+'explainer.html')
    if my_file.is_file():
        st.markdown('The following shows the same output as above but for every instance within a single chart')
        HtmlFile = open(my_file, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        # print(source_code)
        components.html(source_code, height=400)

    st.markdown('The following visualisations show the importance of each attribute of the dataset on the predictions made by the model ')
    st.markdown('The plot below sorts the attributes by the magnitude of the effect it has on the model and the distribution of the impacts each attribute has on the model output. The color represents the feature value (red high, blue low). ')
    my_file = Path(explainabilityPath + 'Summary_Plot.png')
    if my_file.is_file():
        img = amb.load_image(my_file)
        st.image(img, use_column_width=True)
        plt.close()

    my_file = Path(explainabilityPath + 'Heatmap.png')
    if my_file.is_file():
        st.markdown('The following heatmap shows the effect of each attribute for each intance')
        img = amb.load_image(my_file)
        st.image(img, use_column_width=True)
        plt.close()
    return

def app():
    modelType = st.session_state['modelType']
    target_att = st.session_state['target_att']
    projectName = st.session_state['projectName']
    modellingModelPath = st.session_state['modellingModelPath']
    modellingDataPath = st.session_state['modellingDataPath']
    explainabilityPath = st.session_state['explainabilityPath']

    unseen_predictions = pd.read_csv(modellingDataPath + 'UnseenPredictions.csv')
    evaluationData = pd.read_csv(modellingDataPath + 'Evaluation_Data.csv')

    st.title('AutoML ' + modelType + ' - eXplainable AI (XAI)')
    st.markdown(
        ' This page uses SHAP (SHapley Additive exPlanations), a game theoretic approach to explain the output of any machine learning model.')
    st.markdown(
        'It should be noted that the visualisation below use the output from a maximum of the first 50 rows of the unseen data')
    st.markdown('### Model Predictions from unseen data')
    st.dataframe(unseen_predictions)

    instancePath = generate_explainability_charts(evaluationData, projectName, modellingModelPath, explainabilityPath)

    model_explainability(instancePath, explainabilityPath)

