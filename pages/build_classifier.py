import streamlit as st
import streamlit.components.v1 as components
import automlbuilder as amb

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

from pycaret.classification import *
from pycaret.utils import check_metric

def setup_model(trainingData, target_att, activateNormalise, normaliseMethod, activateTransform, transformMethod, fixImbalance, combineLevels, featureInteraction, featureRatio,
                featureSelection):
    projectName = st.session_state.projectName
    envSetup = setup(data=trainingData, target=target_att, session_id=42, train_size=0.9, normalize=activateNormalise, normalize_method=normaliseMethod,
                     transformation=activateTransform, transformation_method=transformMethod, fix_imbalance=fixImbalance, combine_rare_levels=combineLevels,
                     feature_interaction=featureInteraction, feature_ratio=featureRatio, feature_selection=featureSelection, silent=True, log_experiment=True,
                     experiment_name=projectName, log_plots=True, log_data=True)

    environmentData = pull(True)
    #best_model = compare_models(exclude=['catboost'])
    best_model = compare_models()
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
    if 'class_config' not in st.session_state:
        classConfig = False
        st.session_state['class_config'] = classConfig

    if 'class_tuning' not in st.session_state:
        classTuning = False
        st.session_state['class_tuning'] = classTuning

    classConfig = st.session_state['class_config']
    classTuning = st.session_state['class_tuning']

    if dataEdited:
        classTuning = True
        classConfig = True

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
    fixImbalance =  st.session_state.fixImbalance
    combineLevels = st.session_state.combineLevels
    featureInteraction = st.session_state.featureInteraction
    featureRatio = st.session_state.featureRatio
    featureSelection = st.session_state.featureSelection

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
        st.markdown('### Feature Interaction')
        st.markdown(
            'Creates new attributes by interacting (a * b) for all numeric attributes in the dataset ')
        interactionSelect = st.radio('Activate "Feature Interaction"', ('Yes', 'No'), index=1)
        st.markdown('### Feature Ratio')
        st.markdown(
            'Creates new attributes by calculating the ratio (a / b) of all numeric attributes in the dataset ')
        ratioSelect = st.radio('Activate "Feature Ratio"', ('Yes', 'No'), index=1)
        st.markdown('### Fix Imbalance')
        st.markdown('Uses SMOTE (Synthetic Minority Over-sampling Technique) to fix an unequal distribution of the target attribute by creating new synthetic datapoints for the minority class')
        imbalanceSelect = st.radio('Activate "Fix Imbalance"', ('Yes', 'No'), index=1)
        st.markdown('### Feature Selection')
        st.markdown(
            'A subset of features are selected using a combination of various permutation importance techniques including Random Forest, Adaboost and Linear correlation with target variable')
        featureSelect = st.radio('Activate "Feature Selection"', ('Yes', 'No'), index=1)
        st.markdown('### Combine Rare Levels')
        st.markdown('Categorical features created during environment setup are combined when there frequency is below 10%')
        combineSelect = st.radio('Activate "Combine Rare Levels"', ('Yes', 'No'), index=0)
        configSubmit = st.form_submit_button("Submit")
    if configSubmit:
        classConfig = True
    if normaliseMethod != 'none':
        activateNormalise = True
        log_list = amb.update_logging(log_list, 'Classifier Environment Configuration', 'Data Normalisation selected - '+normaliseMethod)
    else:
        activateNormalise = False
        normaliseMethod = 'zscore'
    if transformMethod != 'none':
        activateTransform = True
        log_list = amb.update_logging(log_list, 'Classifier Environment Configuration', 'Data Transformation selected - ' + transformMethod)
    else:
        activateTransform = False
        transformMethod = 'yeo-johnson'
    if interactionSelect != 'No':
        featureInteraction = True
        log_list = amb.update_logging(log_list, 'Classifier Environment Configuration',
                                      'Feature Interaction Activated')
    else:
        featureInteraction = False
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
    if imbalanceSelect != 'No':
        fixImbalance = True
        log_list = amb.update_logging(log_list, 'Classifier Environment Configuration',
                                      'Fix Imbalance Activated')
    else:
        fixImbalance = False
    if featureSelect != 'No':
        featureSelection = True
        log_list = amb.update_logging(log_list, 'Classifier Environment Configuration',
                                      'Feature Selection Activated')
    else:
        featureSelection = False
    if combineSelect != 'Yes':
        combineLevels = False
        log_list = amb.update_logging(log_list, 'Classifier Environment Configuration',
                                      'Combine Rare Levels deactivated')
    else:
        combineLevels = True


    st.session_state.activateNormalise = activateNormalise
    st.session_state.normaliseMethod = normaliseMethod
    st.session_state.activateTransform = activateTransform
    st.session_state.transformMethod = transformMethod
    st.session_state.fixImbalance = fixImbalance
    st.session_state.combineLevels = combineLevels
    st.session_state.featureInteraction = featureInteraction
    st.session_state.featureRatio = featureRatio
    st.session_state.featureSelection = featureSelection

    if not os.path.isfile(modellingDataPath + "Modelling_Environment_Config.txt") or classConfig:
        file = open(modellingDataPath + "Modelling_Environment_Config.txt", "w")
        file.write(str(activateNormalise) + "\n")
        file.write(normaliseMethod + "\n")
        file.write(str(activateTransform) + "\n")
        file.write(transformMethod + "\n")
        file.write(str(interactionSelect)+"\n")
        file.write(str(ratioSelect) + "\n")
        file.write(str(fixImbalance) + "\n")
        file.write(str(combineLevels) + "\n")
        file.write(str(featureSelection) + "\n")
        file.close()

    if not os.path.isfile(modellingDataPath+"Environment_Data.html") or classConfig or classTuning:
        log_list = amb.update_logging(log_list, 'Build Classifier',
                                      'Split dataset, 90 % for modelling, 10 % unseen prediction')
        trainingData, evaluationData = amb.setup_modelling_data(workingData)

        log_list = amb.update_logging(log_list, 'Build Classifier',
                                      'AutoML Environment Setup')
        best_model, train, test, environmentData, modelComparison = setup_model(trainingData, target_att, activateNormalise, normaliseMethod, activateTransform, transformMethod,
                                                                                fixImbalance, combineLevels, featureInteraction, featureRatio, featureSelection)

        st.markdown('### AutoML Environment')
        st.markdown('Table showing the configuration of the modelling environment')
        my_file = Path(modellingDataPath + "Environment_Data.html")

        f = open(my_file, "w")
        style_text = '<style>table {border-collapse: collapse;border-radius: 5px;box-shadow: 0 0 4px rgba(0, 0, 0, 0.25);overflow: hidden;font-family: "Ariel", sans-serif;font-weight: bold;font-size: 14px;}th {background: #4B4B4B;color: #ffffff;text-align: left;}th,td {padding: 10px 20px;}tr:nth-child(even) {background: #eeeeee;}</style>\n'
        f.write(style_text + environmentData.render())
        f.close()

        HtmlFile = open(my_file, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=500, scrolling=True)

    else:
        my_file = Path(modellingDataPath + "Environment_Data.html")
        HtmlFile = open(my_file, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=500, scrolling=True)
    st.markdown('### Transformed Data')
    st.markdown('The data created during the environment setup to be used for developing the model.')
    try:
        st.dataframe(train)
    except:
        st.write('Unable to display training data dataframe at this time')

    st.markdown ('### Transformed Modelling Data Report')
    st.markdown('A Sweetviz report for the data that will be used for training and testing during model development')

    if not os.path.isfile(modellingReportsPath + 'TrainingData_Report.html') or classConfig or classTuning:
        amb.generate_sv(train, '', 'TrainingData',test, True, modellingReportsPath)
        log_list = amb.update_logging(log_list, 'Build Classifier',
                                      'Generating modelling and test data Sweetviz data comparison report')
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

    if not os.path.isfile(modellingDataPath + 'Model_Comparison.html') or classConfig or classTuning:
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

        metricSelect = 'Accuracy'

        if not os.path.isfile(modellingDataPath + 'Model_Comparison.html') or 'modelComparison' in locals():
            modelList = modelComparison.index.tolist()
        else:
            modelComparison = pd.read_csv(modellingDataPath + 'Model_Comparison.csv')
            modelList = modelComparison['Unnamed: 0']

        st.markdown('## Model Configurator - Tuning')
        st.markdown('### Change Model')
        modelChange = st.selectbox('Select model to be used...', modelList)
        st.markdown('### Optimisation Metric')
        metricSelect = st.radio('Select metric used to optimize model during tuning',
                                ('Accuracy', 'AUC', 'Recall', 'F1', 'Kappa', 'MCC'), index=0)
        numIterations = st.slider('Maximum number of tuning iterations...', 10, 50, step=5)
        st.markdown('### Use Ensemble Model')
        st.markdown('Ensembling model is a common technique used for improving performance of a model')
        st.markdown('Bagging - is a machine learning ensemble meta-algorithm designed to improve stability and accuracy of machine learning algorithms. It also reduces variance and helps to avoid overfitting.')
        st.markdown('Boosting - is an ensemble meta-algorithm used primarily for reducing bias and variance in supervised learning. ')
        ensembleSelect = st.radio('Select Ensemble type',('None','Bagging','Boosting'), index=0)
        numEstimators = st.slider('Increase number of estimators used...',10,500, step=10)
        tuningSubmit = st.form_submit_button("Submit")

    if tuningSubmit:
        classTuning = True

    if not os.path.isfile(modellingDataPath + "Tuning_Results.html") or classConfig or classTuning:
        st.markdown('##### Model optimisation metric used = ' + metricSelect)
        log_list = amb.update_logging(log_list, 'Build Classifier',
                                      'Tuning Model - Model optimisation metric  = '+metricSelect)
        st.markdown('##### Maximum number of tuning iterations = ' + str(numIterations))
        log_list = amb.update_logging(log_list, 'Build Classifier',
                                      'Tuning Model - Maximum Number of tuning iterations = ' + str(numIterations))
        if ensembleSelect != 'None':
            st.markdown ('##### Ensemble type = '+ensembleSelect)
            st.markdown ('##### Number of estimators = '+str(numEstimators))
            log_list = amb.update_logging(log_list, 'Build Classifier',
                                          'Tuning Model - Ensemble Type = '+ensembleSelect+' - Number of Estimators = '+str(numEstimators))

        log_list = amb.update_logging(log_list, 'Build Classifier',
                                      'Tuning Model - '+modelChange)
        if classTuning and not classConfig:
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
                    featureInteraction = True
                else:
                    featureInteraction = False
                if str(lines[5].strip()) == 'True':
                    ratioSelect = True
                else:
                    ratioSelect = False
                if str(lines[6].strip()) == 'True':
                    fixImbalance = True
                else:
                    fixImbalance = False
                if str(lines[7].strip()) == 'True':
                    combineLevels = True
                else:
                    combineLevels = False
                if str(lines[8].strip()) == 'True':
                    featureSelection = True
                else:
                    featureSelection = False

            trainingData, evaluationData = amb.setup_modelling_data(workingData)
            best_model, train, test, environmentData, modelComparison = setup_model(trainingData, target_att,
                                                                                    activateNormalise, normaliseMethod,
                                                                                    activateTransform, transformMethod,
                                                                                    fixImbalance, combineLevels,
                                                                                    featureInteraction, featureRatio,
                                                                                    featureSelection)
        tunedModel, tuningData = build_model(ensembleSelect, metricSelect, numEstimators, numIterations, modelChange)

        st.markdown ('### Best Model (Tuned)')
        st.write(tunedModel)
        logModelDescription = str(tunedModel)
        logModelDescription = logModelDescription.replace("\n"," ")
        log_list = amb.update_logging(log_list, 'Build Classifier',
                                      'Best Model - '+logModelDescription)
        file = open(modellingModelPath + "model_description.txt", "w")
        model_desc = repr(tunedModel)
        file.write(model_desc + "\n")
        file.close()
        amb.csv_to_html(tuningData, '#4B4B4B', modellingDataPath, 'Tuning_Results.html')
        st.markdown ('### Model Tuning Results')
        st.markdown('Shows the performance scores from training and testing the model whilst tuning for each fold')
        my_file = Path(modellingDataPath + "Tuning_Results.html")
        if my_file.is_file():
            HtmlFile = open(my_file, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=500, scrolling=True)
    else:
        with open(modellingModelPath + "model_description.txt") as f:
            contents = f.readlines()
        st.markdown('### Best Model (Tuned)')
        st.write(contents)
        st.markdown('### Model Tuning Results')
        my_file = Path(modellingDataPath + "Tuning_Results.html")
        if my_file.is_file():
            HtmlFile = open(my_file, 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=500, scrolling=True)
    if not os.path.isfile(modellingAnalysisPath + "area_under_curve.png") or classConfig or classTuning:
        st.markdown ('### Model Analysis')
        st.markdown ('Selection of plots to be used to evaluate the model')
        X_train = train
        y_train = get_config('y_train')
        X_test = get_config('X_test')
        y_test = get_config('y_test')
        amb.generate_classification_model_analysis(tunedModel, X_train, y_train, X_test, y_test, modellingAnalysisPath)

    filenames = os.listdir(modellingAnalysisPath)
    if filenames:
        filenames.sort()
        selected_filename = st.selectbox('Select a visualisation:', filenames)
        img = amb.load_image(modellingAnalysisPath + selected_filename)
        st.image(img, use_column_width=True)
        plt.close()
    else:
        st.markdown('No analysis plots are available for this model')

    if not os.path.isfile(modellingDataPath + "UnseenPredictions.html") or classConfig or classTuning:
        finalModel = finalize_model(tunedModel)
        unseen_predictions = predict_model(finalModel, data=evaluationData)

        accuracy = check_metric(unseen_predictions[target_att], unseen_predictions['Label'], metric='Accuracy')

        st.markdown('#### Model accuracy on unseen data = '+str(accuracy * 100)+'%')
        log_list = amb.update_logging(log_list, 'Build Classifier', 'Model accuracy on unseen data = '+str(accuracy * 100)+'%')

        save_model(finalModel, modellingModelPath+projectName+'_finalised_model')

        st.markdown('### Unseen Data Predictions')
        st.markdown('The following table shows the unseen data and the predictions made by the final model')
        st.markdown('The predicted value is in the column headed "label", the column headed "score" contains the probability of a positive outcome')
        st.dataframe(unseen_predictions.astype('object'))

        st.markdown('### Unseen Data Predictions - Confusion Matrix')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        try:
            st.pyplot(amb.prediction_confusion_matrix(unseen_predictions, target_att))
        except:
            st.write('This function is only available for binary classification')
        plt.close()

        unseen_predictions.to_csv(modellingDataPath + 'UnseenPredictions.csv', index=False, )
        amb.csv_to_html(unseen_predictions, '#B000FD', modellingDataPath, 'UnseenPredictions.html')
        evaluationData.to_csv(modellingDataPath + 'Evaluation_Data.csv', index=False, )
        amb.csv_to_html(evaluationData, '#FD0000', modellingDataPath, 'Evaluation_Data.html')
    else:
        st.markdown('### Unseen Data Predictions')
        st.markdown('The following table shows the unseen data and the predictions made by the final model')
        st.markdown('Note: The final model used for predicitions is re-trained using all of the modelling data (both training and test data)')
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
        accuracy = check_metric(unseen_predictions[target_att], unseen_predictions['Label'], metric='Accuracy')
        st.markdown('#### Model accuracy on unseen data = ' + str(accuracy * 100) + '%')

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






