import streamlit as st
import pandas as pd
import shap
from pycaret.classification import load_model

import automlbuilder as amb

def model_explainability(data_idx, evaluationData, target_att, modelType, projectName, modellingModelPath):
    evaluationData
    evaluationData.drop([target_att], axis=1, inplace=True)

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

        st.markdown('The folowing shows how each of the features contributes to push the model output from the base value, the average model output over the entire training dataset used to develop the model, to the predicted output of the selected instance of the unseen data.')
        st.markdown('Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.')

        with st.form('instance_select'):
            data_idx = st.slider('Select instance to view from the Unseen Data predictions', 0, maximumRows)
            submitted = st.form_submit_button("Submit")
        st.write('Displaying instance - ' + str(data_idx))
        try:
            amb.st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][data_idx, :], train_pipe.iloc[data_idx, :]))
        except:
            try:
                amb.st_shap(shap.force_plot(explainer.expected_value, shap_values[data_idx, :], train_pipe.iloc[data_idx, :]))
            except:
                st.write('Failed to create SHAP Force Plot')
        st.markdown('The following shows the same output as above but for every instance')
        try:
            amb.st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], train_pipe), height=400)
        except:
            try:
                amb.st_shap(shap.force_plot(explainer.expected_value, shap_values, train_pipe), height=400)
            except:
                st.write('Failed to create SHAP Overall Force Plot')

        #shap_values = explainer.shap_values(X=train_pipe)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown('The following visualisations show the importance of each attribute of the dataset on the predictions made by the model ')
        st.markdown('The plot below sorts the attributes by the magnitude of the effect it has on the model and the distribution of the impacts each attribute has on the model output. The color represents the feature value (red high, blue low). ')
        st.pyplot(shap.summary_plot(shap_values, train_pipe))

    except:
        print('Unable to create TreeExplainer SHAP explainability report')
        try:
            explainer = shap.Explainer(saved_model.named_steps["trained_model"], train_pipe)
            shap_values = explainer(train_pipe)

            st.markdown(
                'The folowing shows how each of the features contributes to push the model output from the base value, the average model output over the entire training dataset used to develop the model, to the predicted output of the selected instance of the unseen data.')
            st.markdown(
                'Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.')
            with st.form('instance_select'):
                data_idx = st.slider('Select instance to view from the Unseen Data predictions', 0, maximumRows)
                submitted = st.form_submit_button("Submit")
            st.write('Displaying instance - ' + str(data_idx))
            st.pyplot(shap.plots.waterfall(shap_values[data_idx]))
#           st.set_option('deprecation.showPyplotGlobalUse', False)
            st.markdown(
                'The following visualisations show the importance of each attribute of the dataset on the predictions made by the model ')
            st.markdown(
                'The plots below sorts the attributes by the magnitude of the effects they have on the model and the distribution of the impacts each attribute has on the model output. The color represents the feature value (red high, blue low). ')
            st.pyplot(shap.plots.beeswarm(shap_values, max_display=20))

            st.pyplot(shap.plots.heatmap(shap_values))

        except:
            print('Unable to create Explainer report')
            try:
                ex = shap.KernelExplainer(saved_model.named_steps["trained_model"].predict, train_pipe)
                shap_values = ex.shap_values(train_pipe.iloc[0, :])

                st.markdown(
                    'The folowing shows how each of the features contributes to push the model output from the base value, the average model output over the entire training dataset used to develop the model, to the predicted output of the selected instance of the unseen data.')
                st.markdown(
                    'Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.')

                with st.form('instance_select'):
                    data_idx = st.slider('Select instance to view from the Unseen Data predictions', 0, maximumRows)
                    submitted = st.form_submit_button("Submit")
                st.write('Displaying instance - ' + str(data_idx))

                st.markdown('The following shows the same output as above but for every instance')
                amb.st_shap(shap.force_plot(ex.expected_value, shap_values, train_pipe.iloc[data_idx, :]))

                st.markdown(
                    'The following visualisations show the importance of each attribute of the dataset on the predictions made by the model ')
                st.markdown(
                    'The plot below sorts the attributes by the magnitude of the effect it has on the model and the distribution of the impacts each attribute has on the model output. The color represents the feature value (red high, blue low). ')
                shap_values = ex.shap_values(X=train_pipe)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(shap.summary_plot(shap_values, train_pipe))
                st.pyplot(shap.summary_plot(shap_values, train_pipe, plot_type='bar'))
            except:
                st.markdown('Unable to generate SHAP explainability report')
    return data_idx

def app():
    modelType = st.session_state['modelType']
    target_att = st.session_state['target_att']
    projectName = st.session_state['projectName']
    modellingModelPath = st.session_state['modellingModelPath']
    modellingDataPath = st.session_state['modellingDataPath']

    unseen_predictions = pd.read_csv(modellingDataPath + 'UnseenPredictions.csv')
    evaluationData = pd.read_csv(modellingDataPath + 'Evaluation_Data.csv')

    st.title('AutoML ' + modelType + ' - Explainability')
    st.markdown(
        ' This page uses SHAP (SHapley Additive exPlanations), a game theoretic approach to explain the output of any machine learning model.')
    st.markdown(
        'It should be noted that the visualisation below use the output from a maximum of the first 50 rows of the unseen data')
    st.markdown('### Model Predictions from unseen data')
    st.dataframe(unseen_predictions)

    if 'data_idx' not in st.session_state:
        st.session_state.data_idx = 0

    data_idx = model_explainability(st.session_state.data_idx, evaluationData, target_att, modelType, projectName, modellingModelPath)

    st.session_state.data_idx = data_idx