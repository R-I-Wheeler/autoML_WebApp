import streamlit as st
import streamlit.components.v1 as components
import automlbuilder as amb

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

from pycaret.regression import *
from pycaret.utils import check_metric

def setup_model(trainingData, target_att, activateNormalise, normaliseMethod, activateTransform, transformMethod,  targetTransform, targetMethod, combineLevels, featureInteraction, featureRatio):

    envSetup = setup(data=trainingData, target=target_att, session_id=42, normalize=activateNormalise, normalize_method=normaliseMethod, transformation=activateTransform, transformation_method=transformMethod,
                         transform_target=targetTransform, transform_target_method=targetMethod, combine_rare_levels=combineLevels, feature_interaction=featureInteraction, feature_ratio=featureRatio, silent=True)
    environmentData = pull(True)
    best_model = compare_models(exclude=['catboost'])
    modelComparison = pull(True)
    train = get_config('X_train')
    test = get_config('X_test')
    return best_model, train, test, environmentData, modelComparison


def build_model(ensembleSelect, metricSelect, numEstimators, numIterations, modelChange):
    model = create_model(modelChange)
    tuned_model = tune_model(model, early_stopping=True, optimize=metricSelect, choose_better=True,
                             n_iter=numIterations)
    if ensembleSelect != 'None':
        built_model = ensemble_model(tuned_model, method=ensembleSelect, optimize=metricSelect, choose_better=True,
                                     n_estimators=numEstimators)
    else:
        built_model = tuned_model
    tuningData = pull(True)
    return built_model, tuningData

def model_development(workingData, target_att, modelType, modellingReportsPath, projectName, modellingDataPath, modellingModelPath, modellingAnalysisPath, log_list, dataEdited):
    if 'regress_config' not in st.session_state:
        regressConfig = False
        st.session_state['regress_config'] = regressConfig

    if 'regress_tuning' not in st.session_state:
        regressTuning = False
        st.session_state['regress_tuning'] = regressTuning

    regressConfig = st.session_state['regress_config']
    regressTuning = st.session_state['regress_tuning']

    if dataEdited:
        regressTuning = True
        regressConfig = True

    st.title('AutoML '+modelType+' - Modelling')
    st.markdown(
        'The data from "Data Analysis" will now be used to develop a machine learning model. The data will first be split, 90% will be used for training and testing the model (split again 70/30) and 10% will be kept back as unseen data.')
    st.markdown(
        'You may find that the number of attributes within the datset has increased, this is because during the setup of the modelling environment the dataset has been analysed and further categorical encoding may have taken place.')
    st.markdown('If you were to use the model developed in this application you would need to use data in the same configuration as outputted at the end of the Data Analysis')
    activateNormalise = st.session_state.activateNormalise
    normaliseMethod = st.session_state.normaliseMethod
    activateTransform = st.session_state.activateTransform
    transformMethod = st.session_state.transformMethod
    targetTransform = st.session_state.targetTransform
    targetMethod = st.session_state.targetMethod
    combineLevels = st.session_state.combineLevels
    featureInteraction = st.session_state.featureInteraction
    featureRatio = st.session_state.featureRatio

    with st.form('environment_config'):
        st.markdown('## AutoML Environment Configuration')
        st.markdown(
            'This section allows you to make further configuration changes to your modelling environment, on completion the setup will run again.')
        st.markdown('### Normalise Data')
        st.markdown('Transforms all numeric attributes by scaling them to a specific range')
        st.markdown('You may select from the following normalisation methods, Z-Score, MinMax, MaxAbs and Robust')
        st.markdown(
            'Z-Score - Calculates the mean of each attribute and then scales the values so that any value equal to the mean is normalised to 0, a value below the mean becomes a negative number and a value above the mean becomes a positive number')
        st.markdown('MinMax - Scales and translates each attribute to a range between 0 and 1')
        st.markdown(
            'MaxAbs - Scales and translates each attribute so that the maximal absolute value of the attribute will be 1.0')
        st.markdown('Robust - scales and translates each attribute according to its Interquartile range')
        normaliseMethod = st.selectbox('Select normalisation method to be used...',
                                       ('none', 'zscore', 'minmax', 'maxabs', 'robust'), 0)
        st.markdown('### Transform Data')
        st.markdown('Makes the data more Gaussian, normalising the distribution. Does not include the target attribute')
        transformMethod = st.selectbox('Select transform method to be used...',
                                       ('none', 'yeo-johnson', 'quantile'), 0)
        st.markdown('### Transform Target')
        st.markdown('Transform the target attribute')
        targetMethod = st.selectbox('Select transform method to be used...',
                                    ('none', 'yeo-johnson', 'box-cox'), 0)
        st.markdown('### Feature Interaction')
        st.markdown(
            'Creates new attributes by interacting (a * b) for all numeric attributes in the dataset ')
        interactionSelect = st.radio('Activate "Feature Interaction"', ('Yes', 'No'), index=1)
        st.markdown('### Feature Ratio')
        st.markdown(
            'Creates new attributes by interacting (a / b) for all numeric attributes in the dataset ')
        ratioSelect = st.radio('Activate "Feature Ratio"', ('Yes', 'No'), index=1)

        st.markdown('### Combine Rare Levels')
        st.markdown('Categorical features are combined when there frequency is below 10%')
        combineSelect = st.radio('Activate "Combine Rare Levels', ('Yes', 'No'), index=0)
        configSubmit = st.form_submit_button("Submit")

    if configSubmit:
        regressConfig = True

    if normaliseMethod != 'none':
        activateNormalise = True
        log_list = amb.update_logging(log_list, 'Regression Environment Configuration',
                                      'Data Normalisation selected - ' + normaliseMethod)
    else:
        activateNormalise = False
        normaliseMethod = 'zscore'
    if transformMethod != 'none':
        activateTransform = True
        log_list = amb.update_logging(log_list, 'Regression Environment Configuration',
                                      'Data Transformation selected - ' + transformMethod)
    else:
        activateTransform = False
        transformMethod = 'yeo-johnson'
    if targetMethod != 'none':
        targetTransform = True
        log_list = amb.update_logging(log_list, 'Regression Environment Configuration',
                                      'Target Transformation selected - ' + targetMethod)
    else:
        targetTransform = False
        targetMethod = 'box-cox'
    if interactionSelect != 'No':
        featureInteraction = True
        log_list = amb.update_logging(log_list, 'Classifier Environment Configuration',
                                      'Feature Interaction Activated')
    else:
        featureInteraction = False
    if ratioSelect != 'No':
        featureRatio = True
        log_list = amb.update_logging(log_list, 'Classifier Environment Configuration',
                                      'Feature Ratio Activated')
    else:
        featureRatio = False
    if combineSelect != 'Yes':
        combineLevels = False
        log_list = amb.update_logging(log_list, 'Regression Environment Configuration',
                                      'Combine Rare Levels deactivated')
    else:
        combineLevels = True

    st.session_state.activateNormalise = activateNormalise
    st.session_state.normaliseMethod = normaliseMethod
    st.session_state.activateTransform = activateTransform
    st.session_state.transformMethod = transformMethod
    st.session_state.targetTransform = targetTransform
    st.session_state.targetMethod = targetMethod
    st.session_state.combineLevels = combineLevels
    st.session_state.featureInteraction = featureInteraction
    st.session_state.featureRatio = featureRatio

    if not os.path.isfile(modellingDataPath + "Modelling_Environment_Config.txt") or regressConfig:
        file = open(modellingDataPath + "Modelling_Environment_Config.txt", "w")
        file.write(str(activateNormalise) + "\n")
        file.write(normaliseMethod + "\n")
        file.write(str(activateTransform) + "\n")
        file.write(transformMethod + "\n")
        file.write(str(targetTransform) + "\n")
        file.write(targetMethod + "\n")
        file.write(str(interactionSelect) + "\n")
        file.write(str(ratioSelect) + "\n")
        file.write(str(combineLevels) + "\n")
        file.close()

    if not os.path.isfile(modellingDataPath+"Environment_Data.html") or regressConfig or regressTuning:
        log_list = amb.update_logging(log_list, 'Build Regression',
                                      'Split dataset, 90 % for modelling, 10 % unseen prediction')
        trainingData, evaluationData = amb.setup_modelling_data(workingData)

        log_list = amb.update_logging(log_list, 'Build Regression',
                                      'AutoML Environment Setup')
        best_model, train, test, environmentData, modelComparison = setup_model(trainingData, target_att,
                                                                                activateNormalise, normaliseMethod,
                                                                                activateTransform, transformMethod,
                                                                                targetTransform, targetMethod,
                                                                                combineLevels, featureInteraction, featureRatio)

        st.markdown('### AutoML Environment')
        st.markdown('Table showing the configuration of the modelling environment')
        st.dataframe(environmentData)
        f = open(modellingDataPath+"Environment_Data.html", "w")
        style_text = '<style>table {border-collapse: collapse;border-radius: 5px;box-shadow: 0 0 4px rgba(0, 0, 0, 0.25);overflow: hidden;font-family: "Ariel", sans-serif;font-weight: bold;font-size: 14px;}th {background: #4B4B4B;color: #ffffff;text-align: left;}th,td {padding: 10px 20px;}tr:nth-child(even) {background: #eeeeee;}</style>\n'
        f.write(style_text+environmentData.render())
        f.close()

        st.markdown ('### Training Data')
        st.markdown('The training data created during the environment setup to be used for developing the model.')
        train = get_config('X_train')
        st.dataframe(train)
    else:
        st.markdown('### AutoML Environment')
        st.markdown('Table showing the configuration of the modelling environment')
        my_file = Path(modellingDataPath+"Environment_Data.html")
        if my_file.is_file():
            HtmlFile = open(my_file, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code, height=500, scrolling=True)


    st.markdown ('### Training & Test Data Comparison')
    st.markdown('A Sweetviz report comparing the data that will be used for training and testing during model development')
    log_list = amb.update_logging(log_list, 'Build Regression',
                                  'Generating modelling and test data Sweetviz data comparison report')

    if not os.path.isfile(modellingReportsPath + 'TrainingData_Report.html') or regressConfig or regressTuning:
        amb.generate_sv(train, '', 'TrainingData',test, True, modellingReportsPath)
    try:
        if os.path.isfile(modellingReportsPath + 'TrainingData_Report.html'):
            HtmlFile = open(modellingReportsPath + 'TrainingData_Report.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code, height=750, scrolling=True)
    except:
        st.write('Unable to display comparison report at this time')

    st.markdown('### Model Comparison')
    st.markdown('A comparison table to evaluate all of the models that have been trained and tested')
    st.markdown(
        'All models within the model library were trained with scores generated using stratified cross validation for metric evaluation')

    if not os.path.isfile(modellingDataPath + 'Model_Comparison.html') or regressConfig or regressTuning:
        st.dataframe(modelComparison)
        amb.csv_to_html(modelComparison, '#4B4B4B', modellingDataPath, 'Model_Comparison.html')
        modelComparison.to_csv(modellingDataPath + 'Model_Comparison.csv')
    else:
        my_file = Path(modellingDataPath + "Model_Comparison.html")
        if my_file.is_file():
            HtmlFile = open(my_file, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code, height=500, scrolling=True)

    with st.form('model_config'):
        numIterations = 10
        numEstimators = 10

        if not os.path.isfile(modellingDataPath + 'Model_Comparison.html') or 'modelComparison' in locals():
            modelList = modelComparison.index.tolist()
        else:
            modelComparison = pd.read_csv(modellingDataPath + 'Model_Comparison.csv')
            modelList = modelComparison['Unnamed: 0']

        st.markdown('## Model Tuning Configuration')
        st.markdown('### Change Model')
        modelChange = st.selectbox('Select model to be used...', modelList)
        st.markdown('### Optimisation Metric')
        metricSelect = st.radio('Select metric used to optimize model during tuning',
                                ('MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'), index=3)
        numIterations = st.slider('Maximum number of tuning iterations...', 10, 50, step=5)
        st.markdown('### Use Ensemble Model')
        st.markdown('Ensembling model is a common technique used for improving performance of a model')
        st.markdown(
            'Bagging - is a machine learning ensemble meta-algorithm designed to improve stability and accuracy of machine learning algorithms. It also reduces variance and helps to avoid overfitting.')
        st.markdown(
            'Boosting - is an ensemble meta-algorithm used primarily for reducing bias and variance in supervised learning. ')
        ensembleSelect = st.radio('Select Ensemble type', ('None', 'Bagging', 'Boosting'), index=0)
        numEstimators = st.slider('Increase number of estimators used...', 10, 500, step=10)
        tuningSubmit = st.form_submit_button("Submit")

    if tuningSubmit:
        regressTuning = True

    if not os.path.isfile(modellingDataPath + "Tuning_Results.html") or regressConfig or regressTuning:
        st.markdown('##### Model optimisation metric used = ' + metricSelect)
        st.markdown('##### Maximum number of tuning iterations = ' + str(numIterations))
        log_list = amb.update_logging(log_list, 'Build Regression',
                                      'Tuning Model - Maximum Number of tuning iterations = ' + str(numIterations))
        log_list = amb.update_logging(log_list, 'Build Regression',
                                      'Tuning Model - Optimisation metric = ' + metricSelect)
        if ensembleSelect != 'None':
            st.markdown ('##### Ensemble type = '+ensembleSelect)
            st.markdown ('##### Number of estimators = '+str(numEstimators))
            log_list = amb.update_logging(log_list, 'Build Regression',
                                          'Tuning Model - Ensemble Type = ' + ensembleSelect + ' - Number of Estimators = ' + str(
                                              numEstimators))
        log_list = amb.update_logging(log_list, 'Build Regression',
                                      'Tuning Model - '+modelChange)
        if regressTuning and not regressConfig:
            with open(modellingDataPath + "Modelling_Environment_Config.txt") as f:
                lines = f.readlines()
                if str(lines[0].strip()) == 'True':
                    activateNormalise = True
                else:
                    activateNormalise = False
                normaliseMethod = str(lines[1].strip())
                if str(lines[2].strip()) == 'True':
                    activateTransform = True
                else:
                    activateTransform = False
                transformMethod = str(lines[3].strip())
                if str(lines[4].strip()) == 'True':
                    targetTransform = True
                else:
                    targetTransform = False
                targetMethod = str(lines[5].strip())
                if str(lines[6].strip()) == 'True':
                    featureInteraction = True
                else:
                    featureInteraction = False
                if str(lines[7].strip()) == 'True':
                    ratioSelect = True
                else:
                    ratioSelect = False
                if str(lines[8].strip()) == 'True':
                    combineLevels = True
                else:
                    combineLevels = False

            trainingData, evaluationData = amb.setup_modelling_data(workingData)
            best_model, train, test, environmentData, modelComparison = setup_model(trainingData, target_att,
                                                                                    activateNormalise, normaliseMethod,
                                                                                    activateTransform, transformMethod,
                                                                                    targetTransform, targetMethod,
                                                                                    combineLevels, featureInteraction, featureRatio)
        tunedModel, tuningData = build_model(ensembleSelect, metricSelect, numEstimators, numIterations, modelChange)

        st.markdown ('### Best Model (Tuned)')
        st.write(tunedModel)
        log_list = amb.update_logging(log_list, 'Build Regression',
                                      'Best Model - '+str(tunedModel))
        file = open(modellingModelPath + "model_description.txt", "w")
        model_desc = repr(tunedModel)
        file.write(model_desc + "\n")
        file.close()

        st.markdown ('### Model Tuning Results')
        st.markdown('Shows the performance scores from training and testing the model whilst tuning for each fold')
        st.write(tuningData)
        amb.csv_to_html(tuningData, '#4B4B4B', modellingDataPath, 'Tuning_Results.html')
    else:
        with open(modellingModelPath + "model_description.txt") as f:
            contents = f.readlines()
        st.markdown('### Best Model (Tuned)')
        st.write(contents)

        st.markdown('### Model Tuning Results')
        st.markdown('Shows the performance scores from training and testing the model whilst tuning for each fold')
        my_file = Path(modellingDataPath + "Tuning_Results.html")
        if my_file.is_file():
            HtmlFile = open(my_file, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code, height=500, scrolling=True)

    if not os.path.isfile(modellingAnalysisPath + "feature_importance.png") or regressConfig or regressTuning:
        st.markdown ('### Model Analysis')
        st.markdown ('Selection of plots to be used to evaluate the model')
        X_train = train
        y_train = get_config('y_train')
        X_test = get_config('X_test')
        y_test = get_config('y_test')
        amb.generate_regression_model_analysis(tunedModel, X_train, y_train, X_test, y_test, modellingAnalysisPath)

    filenames = os.listdir(modellingAnalysisPath)
    if filenames:
        filenames.sort()
        selected_filename = st.selectbox('Select a visualisation:', filenames)
        img = amb.load_image(modellingAnalysisPath + selected_filename)
        st.image(img, use_column_width=True)
        plt.close()
    else:
        st.markdown('No analysis plots are available for this model')

    if not os.path.isfile(modellingDataPath + "UnseenPredictions.html") or regressConfig or regressTuning:
        finalModel = finalize_model(tunedModel)
        unseen_predictions = predict_model(finalModel, data=evaluationData)

        accuracy = check_metric(unseen_predictions[target_att], unseen_predictions['Label'], metric='MAE')

        st.markdown('#### Mean Average Error of predicted values using the unseen data = ' + str(accuracy))
        log_list = amb.update_logging(log_list, 'Build Regression',
                                      'Mean Average Error of predicted values using the unseen data = ' + str(accuracy))

        save_model(finalModel, modellingModelPath+projectName+'_finalised_model')

        st.markdown('### Unseen Data Predictions')
        st.markdown('The following table shows the unseen data and the predictions made by the final model')
        st.markdown(
            'Note: The final model used for predicitions is re-trained using all of the modelling data (both training and test data)')
        st.markdown('The predicted value is in the column headed "label", the column headed "score" contains the probability of a positive outcome')
        st.dataframe(unseen_predictions.astype('object'))

        unseen_predictions.to_csv(modellingDataPath + 'UnseenPredictions.csv', index=False, )
        amb.csv_to_html(unseen_predictions, '#B000FD', modellingDataPath, 'UnseenPredictions.html')
        evaluationData.to_csv(modellingDataPath + 'Evaluation_Data.csv', index=False, )
        amb.csv_to_html(evaluationData, '#FD0000', modellingDataPath, 'Evaluation_Data.html')
    else:
        st.markdown('### Unseen Data Predictions')
        st.markdown('The following table shows the unseen data and the predictions made by the final model')
        st.markdown(
            'The predicted value is in the column headed "label", the column headed "score" contains the probability of a positive outcome')
        my_file = Path(modellingDataPath + "UnseenPredictions.html")
        if my_file.is_file():
            HtmlFile = open(my_file, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            # print(source_code)
            components.html(source_code, height=800, scrolling=True)

        evaluationData = pd.read_csv(modellingDataPath + 'Evaluation_Data.csv')
        finalModel = load_model(modellingModelPath+projectName+'_finalised_model')

        unseen_predictions = predict_model(finalModel, data=evaluationData)
        accuracy = check_metric(unseen_predictions[target_att], unseen_predictions['Label'], metric='MAE')
        st.markdown('#### Mean Average Error of predicted values using the unseen data = ' + str(accuracy))

    return log_list

def app():
    modelType = st.session_state['modelType']
    target_att = st.session_state['target_att']
    projectName = st.session_state['projectName']
    modellingModelPath = st.session_state['modellingModelPath']
    modellingReportsPath = st.session_state['modellingReportsPath']
    modellingDataPath = st.session_state['modellingDataPath']
    modellingAnalysisPath = st.session_state['modellingAnalysisPath']
    log_list = st.session_state['log_list']
    dataEdited = st.session_state['data_edited']

    workingData = pd.read_csv(modellingDataPath+'Modelling_Data.csv')

    log_list = model_development(workingData, target_att, modelType, modellingReportsPath, projectName, modellingDataPath,
                                       modellingModelPath, modellingAnalysisPath, log_list, dataEdited)

    st.session_state['data_edited'] = False
    st.session_state['log_list'] = log_list

    plt.clf()






