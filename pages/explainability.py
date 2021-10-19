import streamlit as st
import pandas as pd
import shap
from pycaret.classification import load_model

import automlbuilder as amb

def model_explainability(unseen_predictions, evaluationData, target_att, modelType, projectName, modellingModelPath):
    st.title('AutoML ' + modelType + ' - Explainability')
    st.markdown(' This page uses SHAP (SHapley Additive exPlanations), a game theoretic approach to explain the output of any machine learning model.')
    st.markdown('It should be noted that the visualisation below use the output from a maximum of the first 50 rows of the unseen data')
    st.markdown('### Model Predictions from unseen data')
    st.dataframe (unseen_predictions)

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

    if 'data_idx' not in st.session_state:
        st.session_state.data_idx = 0

    data_idx = st.session_state['data_idx']
    try:
        explainer = shap.TreeExplainer(saved_model.named_steps["trained_model"])
        shap_values = explainer.shap_values(X=train_pipe.iloc[0:maximumRows, :])

        st.markdown('The folowing shows how each of the features contributes to push the model output from the base value, the average model output over the entire training dataset used to develop the model, to the predicted output of the selected instance of the unseen data.')
        st.markdown('Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.')
        amb.st_shap(shap.force_plot(explainer.expected_value, shap_values[data_idx, :], train_pipe.iloc[data_idx, :]))
        data_idx = st.slider('Select instance to view from the Unseen Data predictions', 0, maximumRows)

        st.markdown('The following shows the same output as above but for every instance')
        amb.st_shap(shap.force_plot(explainer.expected_value, shap_values, train_pipe), height=400)

        shap_values = explainer.shap_values(X=train_pipe)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown('The following visualisations show the importance of each attribute of the dataset on the predictions made by the model ')
        st.markdown('The plot below sorts the attributes by the magnitude of the effect it has on the model and the distribution of the impacts each attribute has on the model output. The color represents the feature value (red high, blue low). ')
        st.pyplot(shap.summary_plot(shap_values, train_pipe))
        st.pyplot(shap.summary_plot(shap_values, train_pipe, plot_type='bar'))

        st.session_state.data_idx = data_idx

    except:
        print('Unable to create TreeExplainer SHAP explainability report')
        try:
            ex = shap.KernelExplainer(saved_model.named_steps["trained_model"].predict, train_pipe)
            shap_values = ex.shap_values(train_pipe.iloc[0:maximumRows, :])

            if modelType == 'Regression':
                st.markdown(
                    'The folowing shows how each of the features contributes to push the model output from the base value, the average model output over the entire training dataset used to develop the model, to the predicted output of the selected instance of the unseen data.')
                st.markdown(
                    'Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.')
                amb.st_shap(shap.force_plot(ex.expected_value, shap_values, train_pipe.iloc[data_idx, :]))
                data_idx = st.slider('Select instance to view from the Unseen Data predictions', 0, maximumRows,
                                         key='kernel_slider')

            shap_values = ex.shap_values(X=train_pipe)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.markdown(
                'The following visualisations show the importance of each attribute of the dataset on the predictions made by the model ')
            st.markdown(
                'The plot below sorts the attributes by the magnitude of the effect it has on the model and the distribution of the impacts each attribute has on the model output. The color represents the feature value (red high, blue low). ')
            st.pyplot(shap.summary_plot(shap_values, train_pipe))
            st.pyplot(shap.summary_plot(shap_values, train_pipe, plot_type='bar'))

     #       st.session_state.data_idx = data_idx
        except:
            st.markdown('Unable to create SHAP explainability report')
    return

def app():
    modelType = st.session_state['modelType']
    target_att = st.session_state['target_att']
    projectName = st.session_state['projectName']
    modellingModelPath = st.session_state['modellingModelPath']
    modellingDataPath = st.session_state['modellingDataPath']

    unseen_predictions = pd.read_csv(modellingDataPath + 'UnseenPredictions.csv')
    evaluationData = pd.read_csv(modellingDataPath + 'Evaluation_Data.csv')

    model_explainability(unseen_predictions, evaluationData, target_att, modelType, projectName, modellingModelPath)
