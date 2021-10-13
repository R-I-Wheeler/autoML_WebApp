import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
import sweetviz as sv
from pandas_profiling import ProfileReport
import klib
from PIL import Image
import os
import shutil
from pandas.api.types import is_numeric_dtype
from pycaret.utils import check_metric
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
from scipy import stats
#import lux

# Function to Read and Manipulate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

@st.cache
def generate_sv(data, target_att, reportname, compareData, compare):
    if compare == False:
        my_report = sv.analyze(data, target_feat=target_att)
    else:
        my_report = sv.compare(data, compareData)
    my_report.show_html(filepath='SWEETVIZ_REPORT.html',
                        open_browser=False,
                        layout='vertical',
                        scale=None)
    os.rename('SWEETVIZ_REPORT.html', reportname+'_Report.html')
    HtmlFile = open(reportname+'_Report.html', 'r', encoding='utf-8')
    sv_report = HtmlFile.read()
    return sv_report

@st.cache
def generate_pp(data):
    try:
        profile = ProfileReport(data, title="Data Profiling Report", explorative=True)
        profile.to_file("pp_OriginalData_Report.html")
        HtmlFile = open('pp_OriginalData_Report.html', 'r', encoding='utf-8')
        pp_report = HtmlFile.read()
        reportSuccess=True
    except:
        reportSuccess=False
        pp_report=None
    return pp_report, reportSuccess

def clean_int(data):
    intEightColumns = data.dtypes[(data.dtypes == np.int8)]
    intEight = list(intEightColumns.index)
    intSixteenColumns = data.dtypes[(data.dtypes == np.int16)]
    intSixteen = list(intSixteenColumns.index)
    intColumnNames = intEight + intSixteen
    data[intColumnNames] = data[intColumnNames].astype('int32')
    return data

@st.cache(allow_output_mutation=True)
def generate_av(data, target_att):
    fig = plt.figure()
    fig.patch.set_facecolor('#E0E0E0')
    fig.patch.set_alpha(0.7)

    data = AV.AutoViz(filename='', dfte=data, depVar=target_att, verbose=2, chart_format='jpg')
    return data

@st.cache(allow_output_mutation=True)
def generate_distPlot(data, target_att):
    cols = data.select_dtypes([np.number]).columns
    for col in cols:
        distplot = klib.dist_plot(data[col])
        if distplot != None:
            distplot.figure.savefig('./AutoViz_Plots/'+target_att+'/'+col+'_Distribution_Plot.png', dpi=100)
    catplot = klib.cat_plot(data, figsize=(30, 30))
    if catplot != None:
        catplot.figure.savefig('./AutoViz_Plots/' + target_att + '/Categorical_Plot.png', dpi=100)
    plt.clf()
    plt.cla()
    return

def AV_file_selector(target_att):
    folder_path = './AutoViz_Plots/'+target_att
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a visualisation:', filenames)
    return os.path.join(folder_path, selected_filename)

def MA_file_selector(target_att):
    folder_path = './modelAnalysis/'+target_att
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a visualisation:', filenames)
    return os.path.join(folder_path, selected_filename)

def drop_numerical_outliers(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)

@st.cache(allow_output_mutation=True)
def clean_data(allData):
    data = klib.data_cleaning(allData, drop_threshold_cols=0.5)
    data = clean_int(data)
    return data

@st.cache
def setupModellingData(data):
    # Split dataset, 90% for modelling, 10% unseen prediction
    trainingData = data.sample(frac=0.9, random_state=42)
    evaluationData = data.drop(trainingData.index)
    trainingData.reset_index(drop=True, inplace=True)
    evaluationData.reset_index(drop=True, inplace=True)
    return trainingData, evaluationData

@st.cache(allow_output_mutation=True)
def setupModel(trainingData, target_att, modelType, transformTarget):
    if modelType == 'Classification':
        envSetup = setup(data=trainingData, target=target_att, session_id=42, feature_selection=True,
                         remove_outliers=True, combine_rare_levels=True, silent=True)
    else:
        if transformTarget == False:
            envSetup = setup(data=trainingData, target=target_att, session_id=42, feature_selection=True,
                             remove_outliers=True, combine_rare_levels=True, silent=True)
        else:
            envSetup = setup(data=trainingData, target=target_att, session_id=42, feature_selection=True,
                             remove_outliers=True, transform_target=True, combine_rare_levels=True, silent=True)
    environmentData = pull(True)
    best_model = compare_models()
    modelComparison = pull(True)
    train = get_config('X_train')
    test = get_config('X_test')
    return best_model, train, test, environmentData, modelComparison


def buildModel(best_model, ensembleSelect, metricSelect, numEstimators, numIterations,modelChange):
    model = create_model(modelChange)
    if ensembleSelect != 'None':
        tuned_model = ensemble_model(model, method=ensembleSelect, optimize=metricSelect, choose_better=True,
                                     n_estimators=numEstimators)
    else:
        tuned_model = tune_model(model, early_stopping=True, optimize=metricSelect, choose_better=True,
                                 n_iter=numIterations)
    tuningData = pull(True)
    return tuned_model, tuningData


def runpredictions(model, evaluationData, modelType):
    final_model = finalize_model(model)
    unseen_predictions = predict_model(final_model, data=evaluationData)
    if modelType == 'Classification':
        accuracy = check_metric(unseen_predictions[target_att], unseen_predictions['Label'], metric='Accuracy')
    else:
        accuracy = check_metric(unseen_predictions[target_att], unseen_predictions['Label'], metric='MAE')
    return final_model,unseen_predictions, accuracy

@st.cache
def luxOutput(data,target_att):
    luxAllData = data.save_as_html(output=True)
    data.intent = [target_att]
    luxTarget = data.save_as_html(output=True)

    #Add code below main to display widgets
    # luxAllData, luxTarget = luxOutput(data,target_att)
    # components.html(luxAllData, width=1100, height=350, scrolling=True)
    # components.html(luxTarget, width=1100, height=350, scrolling=True)
    return luxAllData, luxTarget

def predictionCM(unseen_predictions):
    cm = confusion_matrix(unseen_predictions[target_att], unseen_predictions['Label'])
    cm_obj = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
    cm_obj.plot()
    cm_obj.ax_.set(
        title='Unseen Data Confusion Matrix',
        xlabel='Predicted',
        ylabel='Actual')
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def modelAnalysisCharts(model, target_att):
    #path = './modelAnalysis/'+target_att
    # Check whether the specified path exists or not
    #isExist = os.path.exists(path)
    #if not isExist:
        # Create a new directory because it does not exist
    #    os.makedirs(path)

    #    print("The new directory is created!")
    if modelType=='Classification':
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
    else:
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
    #sourcepath = './'
    #sourcefiles = os.listdir(sourcepath)
    #destinationpath = path
    #for file in sourcefiles:
    #    if file.endswith('.png'):
    #        shutil.move(os.path.join(sourcepath, file), os.path.join(destinationpath, file))
    return

def analysis(data, target_att, modelType, transformTarget):
    st.title('AutoML ' + modelType + ' - Data Analysis')

    st.markdown('## Original Dataset')
    st.dataframe(data.astype('object'))

    st.markdown('## Data Report')
    dataReporting = st.selectbox('Select data report:', ('Pandas Profile','Sweetviz'))
    if dataReporting == 'Sweetviz':
        components.html(sv_report, width=1000,height=1000, scrolling=True)
    else:
        if reportSuccess == True:
            components.html(pp_report, width=1000, height=1000, scrolling=True)
        else:
            st.write ('Failed to generate Pandas Profile report')

    st.markdown('## Quick Data Visualisation')

    filename = AV_file_selector(target_att)
    img = load_image(filename)
    st.image(img,use_column_width=True)
    plt.close()

    with st.form('drop_columns'):
        st.markdown('## Data Configurator')
        st.markdown('#### Select columns to drop from dataset...')
        dataColumns = list(data.columns)
        dataColumns.remove(target_att)
        dropSelect = st.multiselect('Select columns...',dataColumns, key='drop_Columns')
        dataColumns = list(data.columns)
        st.markdown('#### Select columns to apply log transformation (numeric only)...')
        normaliseSelect = st.multiselect('Select columns...', dataColumns, key='normalise_Columns')
        outlierSelect = st.radio('Remove Outliers from numeric data',('Yes','No'), index=1)
        submitted = st.form_submit_button("Submit")
        if submitted:
            if dropSelect != None:
                data.drop(dropSelect, axis=1, inplace=True)
            if normaliseSelect != None:
                for col in normaliseSelect:
                    if is_numeric_dtype(data[col]):
                        if col == target_att:
                            transformTarget = True
                        else:
                            data[col] = np.log1p(data[col])
            if outlierSelect == 'Yes':
                drop_numerical_outliers(data)
    return (data, transformTarget)

def modelling(data,target_att, modelType, transformTarget):
    st.title('AutoML '+modelType+' - Modelling')
    trainingData, evaluationData = setupModellingData(data)
    best_model, train, test, environmentData, modelComparison = setupModel(trainingData, target_att, modelType, transformTarget)


    st.markdown ('### Training Data')
    train = get_config('X_train')
    st.dataframe(train)

    st.markdown ('### Training & Test Data Comparison')
    training_report = generate_sv(train, '', 'TrainingData',test, True)
    components.html(training_report, width=1000, height=1000, scrolling=True)

    st.markdown('### AutoML Environment')
    st.dataframe(environmentData)

    st.markdown('### Model Comparison')
    st.dataframe(modelComparison)

    with st.form('model_config'):
        numIterations = 10
        numEstimators = 10
        ensembleSelect = 'None'
        if modelType == 'Classification':
            metricSelect = 'Accuracy'
        else:
            metricSelect = 'R2'
        modelList = modelComparison.index.tolist()
        modelChange = modelList[0]
        st.markdown('## Model Configurator - Tuning')
        st.markdown('### Change Model')
        modelChange = st.selectbox('Select model to be used...', modelList)
        st.markdown('### Use Ensemble Model')
        ensembleSelect = st.radio('Select Ensemble type',('None','Bagging','Boosting'), index=0)
        if ensembleSelect != 'None':
            numEstimators = st.slider('Increase number of estimators used...',10,500, step=10)
        st.markdown('### Optimisation Metric')
        if modelType == 'Classification':
            metricSelect = st.radio('Select metric used to optimize model during tuning',
                                      ('Accuracy', 'AUC', 'Recall', 'F1', 'Kappa', 'MCC'), index=0)
        else:
            metricSelect = st.radio('Select metric used to optimize model during tuning',
                                      ('MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'), index=3)
        if ensembleSelect == 'None':
            numIterations = st.slider('Maximum number of tuning iterations...',10,50, step=5)
        submitted = st.form_submit_button("Submit")
    #modelUse
    st.markdown('##### Model optimisation metric used = ' + metricSelect)
    if ensembleSelect != 'None':
        st.markdown ('##### Ensemble type = '+ensembleSelect)
        st.markdown ('##### Number of estimators = '+str(numEstimators))
    else:
        st.markdown ('##### Maximum number of tuning iterations = '+str(numIterations))

    tunedModel, tuningData = buildModel(best_model, ensembleSelect, metricSelect, numEstimators, numIterations, modelChange)

    st.markdown ('### Best Model (Tuned)')
    st.write(tunedModel)

    st.markdown ('### Model Tuning Results')
    st.write(tuningData)

    st.markdown ('### Model Analysis')
    modelAnalysisCharts(tunedModel, target_att)
    #maFilename = MA_file_selector(target_att)
    #img = load_image(maFilename)
    #st.image(img, use_column_width=True)
    #plt.close()

    finalModel, unseen_predictions, accuracy = runpredictions(tunedModel,evaluationData, modelType)
    if modelType == 'Classification':
        st.markdown ('### Unseen Data Predictions - Confusion Matrix')
        predictionCM(unseen_predictions)
        st.markdown('#### Model accuracy on unseen data = '+str(accuracy * 100)+'%')
    else:
        st.markdown('#### Mean Average Error of predicted values = ' + str(accuracy))

    save_model(finalModel, 'finalised_model')

    st.markdown('### Unseen Data Predictions')
    st.dataframe(unseen_predictions.astype('object'))
    return (evaluationData)

def explainability(evaluationData, target_att):
    st.title('AutoML ' + modelType + ' - Explainability')
    explainData = evaluationData.copy()
    explainData.drop([target_att], axis=1, inplace=True)

    saved_model = load_model('finalised_model')
    train_pipe = saved_model[:-1].transform(explainData)

    numRows = train_pipe[train_pipe.columns[0]].count()
    numRows.item()

    if numRows <= 50:
        maxRows = numRows
    else:
        maxRows = 50
    maxRows = getattr(maxRows, "tolist", lambda: maxRows)()

    try:
        explainer = shap.TreeExplainer(saved_model.named_steps["trained_model"])
        shap_values = explainer.shap_values(X=train_pipe.iloc[0:maxRows, :])

        data_idx = 0
        data_idx = st.slider(
            'Select instance to view from the first ' + str(maxRows) + ' rows from "Unseen Data Predictions" ', 0, maxRows)
        st_shap(shap.force_plot(explainer.expected_value, shap_values[data_idx, :], train_pipe.iloc[data_idx, :]))

        st_shap(shap.force_plot(explainer.expected_value, shap_values, train_pipe), height=400)

        shap_values = explainer.shap_values(X=train_pipe)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.summary_plot(shap_values, train_pipe))
        st.pyplot(shap.summary_plot(shap_values, train_pipe, plot_type='bar'))

    except:
        st.markdown('Unable to create SHAP explainability report')
        try:
            data_idx = 0
            ex = shap.KernelExplainer(saved_model.named_steps["trained_model"].predict, train_pipe)
            shap_values = ex.shap_values(train_pipe.iloc[0, :])

            data_idx = st.slider(
                'Select instance to view from the first ' + str(maxRows) + ' rows from "Unseen Data Predictions" ', 0,
                maxRows, key='kernel_slider')
            st_shap(shap.force_plot(ex.expected_value, shap_values, train_pipe.iloc[data_idx, :]))

            shap_values = ex.shap_values(X=train_pipe)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.summary_plot(shap_values, train_pipe))
            st.pyplot(shap.summary_plot(shap_values, train_pipe, plot_type='bar'))
        except:
            st.markdown('Unable to create SHAP explainability report')
    return


st.sidebar.header("Load Data")
uploaded_file = st.sidebar.file_uploader("Choose a csv file", type='csv')
if uploaded_file is not None:
    allData = pd.read_csv(uploaded_file)
    modelType = st.sidebar.selectbox('Select Model Type:', ('Classification', 'Regression'))
    data = clean_data(allData)
    column_list = list(data.columns)
    column_list.insert(0,'Select Target Attribute...')
    target_att = st.sidebar.selectbox('Select Target Attribute:', (column_list))

    if target_att != 'Select Target Attribute...':
        if modelType == 'Regression':
            from pycaret.regression import *
        else:
            from pycaret.classification import *
            data[target_att] = data[target_att].astype('int32')

        transformTarget = False
        data = generate_av(data, target_att)
        #generate_distPlot(data, target_att)
        sv_report = generate_sv(data, target_att, 'OriginalData', None, False)
        pp_report, reportSuccess = generate_pp(data)

        data,transformTarget = analysis(data,target_att, modelType, transformTarget)
        #plt.cla()
        plt.clf()
        evaluationData = modelling(data,target_att, modelType, transformTarget)
        #plt.cla()
        plt.clf()
        explainability(evaluationData, target_att)
else:
    st.stop()




