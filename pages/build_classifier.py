import streamlit as st
import streamlit.components.v1 as components
import automlbuilder as amb
import matplotlib.pyplot as plt

from pycaret.classification import *
from pycaret.utils import check_metric


def setup_model(trainingData, target_att):

    envSetup = setup(data=trainingData, target=target_att, session_id=42, feature_selection=True,
                     combine_rare_levels=True, silent=True)

    environmentData = pull(True)
    best_model = compare_models()
    modelComparison = pull(True)
    train = get_config('X_train')
    test = get_config('X_test')
    return best_model, train, test, environmentData, modelComparison


def build_model(ensembleSelect, metricSelect, numEstimators, numIterations, modelChange):
    model = create_model(modelChange)
    if ensembleSelect != 'None':
        tuned_model = ensemble_model(model, method=ensembleSelect, optimize=metricSelect, choose_better=True,
                                     n_estimators=numEstimators)
    else:
        tuned_model = tune_model(model, early_stopping=True, optimize=metricSelect, choose_better=True,
                                 n_iter=numIterations)
    tuningData = pull(True)
    return tuned_model, tuningData

def model_analysis_charts(model, modelType):
    modelAnalysis = st.selectbox('Select model analysis plot:', ('Confusion Matrix', 'Classification Report',
                                                                 'Class Prediction Error', 'Learning Curve',
                                                                 'Area Under the Curve'))
    if modelAnalysis == 'Confusion Matrix':
        plot_model(model, plot='confusion_matrix', display_format='streamlit')
    elif modelAnalysis == 'Classification Report':
        plot_model(model, plot='class_report', display_format='streamlit')
    elif modelAnalysis == 'Class Prediction Error':
        plot_model(model, plot='error', display_format='streamlit')
    elif modelAnalysis == 'Learning Curve':
        plot_model(model, plot='learning', display_format='streamlit')
    elif modelAnalysis == 'Area Under the Curve':
        plot_model(model, plot='auc', display_format='streamlit')
    return

def model_development(workingData, target_att, modelType, modellingReportsPath, projectName, modellingDataPath, modellingModelPath):

    st.title('AutoML '+modelType+' - Modelling')
    trainingData, evaluationData = amb.setup_modelling_data(workingData)
    best_model, train, test, environmentData, modelComparison = setup_model(trainingData, target_att)


    st.markdown ('### Training Data')
    train = get_config('X_train')
    st.dataframe(train)

    st.markdown ('### Training & Test Data Comparison')
    training_report = amb.generate_sv(train, '', 'TrainingData',test, True,modellingReportsPath)
    try:
        components.html(training_report, width=1000, height=1000, scrolling=True)
    except:
        print('Unable to display comparison report at this time')
        try:
            HtmlFile = open(modellingReportsPath+"TrainingData_Report.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=1000, scrolling=True)
        except:
            st.write('Unable to display comparison report at this time')


    st.markdown('### AutoML Environment')
    st.dataframe(environmentData)

    st.markdown('### Model Comparison')
    st.dataframe(modelComparison)

    with st.form('model_config'):
        numIterations = 10
        numEstimators = 10

        metricSelect = 'Accuracy'

        modelList = modelComparison.index.tolist()

        st.markdown('## Model Configurator - Tuning')
        st.markdown('### Change Model')
        modelChange = st.selectbox('Select model to be used...', modelList)
        st.markdown('### Use Ensemble Model')
        ensembleSelect = st.radio('Select Ensemble type',('None','Bagging','Boosting'), index=0)
        if ensembleSelect != 'None':
            numEstimators = st.slider('Increase number of estimators used...',10,500, step=10)
        st.markdown('### Optimisation Metric')
        metricSelect = st.radio('Select metric used to optimize model during tuning',
                                  ('Accuracy', 'AUC', 'Recall', 'F1', 'Kappa', 'MCC'), index=0)
        if ensembleSelect == 'None':
            numIterations = st.slider('Maximum number of tuning iterations...',10,50, step=5)
        submitted = st.form_submit_button("Submit")

    st.markdown('##### Model optimisation metric used = ' + metricSelect)
    if ensembleSelect != 'None':
        st.markdown ('##### Ensemble type = '+ensembleSelect)
        st.markdown ('##### Number of estimators = '+str(numEstimators))
    else:
        st.markdown ('##### Maximum number of tuning iterations = '+str(numIterations))

    tunedModel, tuningData = build_model(ensembleSelect, metricSelect, numEstimators, numIterations, modelChange)

    st.markdown ('### Best Model (Tuned)')
    st.write(tunedModel)

    st.markdown ('### Model Tuning Results')
    st.write(tuningData)

    st.markdown ('### Model Analysis')
    model_analysis_charts(tunedModel, modelType)

    finalModel = finalize_model(tunedModel)
    unseen_predictions = predict_model(finalModel, data=evaluationData)

    accuracy = check_metric(unseen_predictions[target_att], unseen_predictions['Label'], metric='Accuracy')


    st.markdown ('### Unseen Data Predictions - Confusion Matrix')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    try:
        st.pyplot(amb.prediction_confusion_matrix(unseen_predictions, target_att))
    except:
        st.write('This function is only available for binary classification')
    st.markdown('#### Model accuracy on unseen data = '+str(accuracy * 100)+'%')

    save_model(finalModel, modellingModelPath+projectName+'_finalised_model')

    st.markdown('### Unseen Data Predictions')
    st.dataframe(unseen_predictions.astype('object'))

    evaluationData.to_csv(modellingDataPath + 'Evaluation_Data.csv', index=False, )
    return


def app():
    modelType = st.session_state['modelType']
    target_att = st.session_state['target_att']
    projectName = st.session_state['projectName']
    modellingModelPath = st.session_state['modellingModelPath']
    modellingReportsPath = st.session_state['modellingReportsPath']
    modellingDataPath = st.session_state['modellingDataPath']

    workingData = pd.read_csv(modellingDataPath+'Modelling_Data.csv')

    model_development(workingData, target_att, modelType, modellingReportsPath, projectName, modellingDataPath,
                                       modellingModelPath)
    plt.clf()



