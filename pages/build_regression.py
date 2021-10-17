import streamlit as st
import streamlit.components.v1 as components
import automlbuilder as amb
import matplotlib.pyplot as plt
import shap
from pycaret.regression import *
from pycaret.utils import check_metric


def setup_model(trainingData, target_att, transformTarget):
    if transformTarget == False:
        envSetup = setup(data=trainingData, target=target_att, session_id=42, feature_selection=True,
                         combine_rare_levels=True, silent=True)
    else:
        envSetup = setup(data=trainingData, target=target_att, session_id=42, feature_selection=True,
                         transform_target=True, combine_rare_levels=True, silent=True)
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

def model_analysis_charts(model):
    modelAnalysis = st.selectbox('Select model analysis plot:', ('Residuals Plot', 'Prediction Error Plot',
                                                                 'Validation Curve', 'Learning Curve'))
    if modelAnalysis == 'Residuals Plot':
        plot_model(model, plot='residuals', display_format='streamlit')
    elif modelAnalysis == 'Prediction Error Plot':
        plot_model(model, plot='error', display_format='streamlit')
    elif modelAnalysis == 'Validation Curve':
        plot_model(model, plot='vc', display_format='streamlit')
    elif modelAnalysis == 'Learning Curve':
        plot_model(model, plot='learning', display_format='streamlit')
    return

def model_development(workingData, target_att, modelType, transformTarget, modellingReportsPath, projectName, modellingDataPath,
                      modellingModelPath):

    st.title('AutoML '+modelType+' - Modelling')
    trainingData, evaluationData = amb.setup_modelling_data(workingData)
    best_model, train, test, environmentData, modelComparison = setup_model(trainingData, target_att, transformTarget)


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
                                  ('MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'), index=3)
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
    model_analysis_charts(tunedModel)

    finalModel = finalize_model(tunedModel)
    unseen_predictions = predict_model(finalModel, data=evaluationData)

    accuracy = check_metric(unseen_predictions[target_att], unseen_predictions['Label'], metric='MAE')

    st.markdown('#### Mean Average Error of predicted values = ' + str(accuracy))

    save_model(finalModel, modellingModelPath+projectName+'_finalised_model')

    st.markdown('### Unseen Data Predictions')
    st.dataframe(unseen_predictions.astype('object'))
    evaluationData.to_csv(modellingDataPath + 'Evaluation_Data.csv', index=False, )
    return

def model_explainability(evaluationData, target_att, modelType, projectName, modellingModelPath):
    st.title('AutoML ' + modelType + ' - Explainability')
    explainData = evaluationData.copy()
    explainData.drop([target_att], axis=1, inplace=True)

    saved_model = load_model(modellingModelPath+projectName+'_finalised_model')
    train_pipe = saved_model[:-1].transform(explainData)

    numRows = train_pipe[train_pipe.columns[0]].count()
    numRows.item()

    if numRows <= 50:
        maximumRows = numRows
    else:
        maximumRows = 50
    maximumRows = getattr(maximumRows, "tolist", lambda: maximumRows)()
    data_idx = 0

    try:
        explainer = shap.TreeExplainer(saved_model.named_steps["trained_model"])
        shap_values = explainer.shap_values(X=train_pipe.iloc[0:maximumRows, :])

        amb.st_shap(shap.force_plot(explainer.expected_value, shap_values[data_idx, :], train_pipe.iloc[data_idx, :]))
        data_idx = st.slider('Select instance to view from the Unseen Data predictions', 0, maximumRows)

        amb.st_shap(shap.force_plot(explainer.expected_value, shap_values, train_pipe), height=400)

        shap_values = explainer.shap_values(X=train_pipe)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.summary_plot(shap_values, train_pipe))
        st.pyplot(shap.summary_plot(shap_values, train_pipe, plot_type='bar'))

    except:
        print('Unable to create TreeExplainer SHAP explainability report')
        try:
            ex = shap.KernelExplainer(saved_model.named_steps["trained_model"].predict, train_pipe)
            shap_values = ex.shap_values(train_pipe.iloc[0:maximumRows, :])


            data_idx = st.slider('Select instance to view from the Unseen Data predictions', 0, maximumRows, key='kernel_slider')
            amb.st_shap(shap.force_plot(ex.expected_value, shap_values, train_pipe.iloc[data_idx, :]))

            shap_values = ex.shap_values(X=train_pipe)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.summary_plot(shap_values, train_pipe))
            st.pyplot(shap.summary_plot(shap_values, train_pipe, plot_type='bar'))
        except:
            st.markdown('Unable to create SHAP explainability report')
    return
target_att = None

def app():
    modelType = st.session_state['modelType']
    target_att = st.session_state['target_att']
    projectName = st.session_state['projectName']
    modellingModelPath = st.session_state['modellingModelPath']
    modellingReportsPath = st.session_state['modellingReportsPath']
    modellingDataPath = st.session_state['modellingDataPath']
    transformTarget = st.session_state['transformTarget']

    workingData = pd.read_csv(modellingDataPath + 'Modelling_Data.csv')

    model_development(workingData, target_att, modelType, transformTarget, modellingReportsPath, projectName,
                                       modellingDataPath, modellingModelPath)

    plt.clf()
    #model_explainability(evaluationData, target_att, modelType, projectName, modellingModelPath)

