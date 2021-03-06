import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
#from autoviz.AutoViz_Class import AutoViz_Class
#AV = AutoViz_Class()
import sweetviz as sv
from pandas_profiling import ProfileReport
import klib
from PIL import Image
import os
import math
import shutil
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
import base64
#import lux
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.model_selection import LearningCurve
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError

# Function to Read and Manipulate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def create_directory(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    return


def generate_sv(workingData, target_att, reportname, compareData, compare, reportsPath):
    try:
        if compare == False:
            my_report = sv.analyze(workingData, target_feat=target_att)
        else:
            my_report = sv.compare(workingData, compareData)
        my_report.show_html(filepath= reportsPath+'SWEETVIZ_REPORT.html',
                            open_browser=False,
                            layout='vertical',
                            scale=None)
        os.rename(reportsPath+'SWEETVIZ_REPORT.html', reportsPath+reportname+'_Report.html')
    except:
        print('Failed to Generate Sweetviz Report')
    return


def generate_pp(workingData, reportsPath):
    try:
        profile = ProfileReport(workingData, title="Data Profiling Report", explorative=True)
        profile.to_file(reportsPath+"pp_OriginalData_Report.html")
    except:
        print('Failed to Generate Pandas Profile Report')
    return

def clean_data(allData):
    drop_col = [e for e in allData.columns if allData[e].nunique() == allData.shape[0]]
    new_df_columns = [e for e in allData.columns if e not in drop_col]
    workingData = allData[new_df_columns]

    workingData = klib.data_cleaning(workingData, drop_threshold_cols=0.5)

    intEightColumns = workingData.dtypes[(workingData.dtypes == np.int8)]
    intEight = list(intEightColumns.index)
    intSixteenColumns = workingData.dtypes[(workingData.dtypes == np.int16)]
    intSixteen = list(intSixteenColumns.index)
    intColumnNames = intEight + intSixteen
    workingData[intColumnNames] = workingData[intColumnNames].astype('int32')

    workingData = workingData.replace(' ', '-', regex=True)
    workingData = workingData.applymap(lambda s: s.upper() if type(s) == str else s)
    return workingData


def half_masked_corr_heatmap(workingdata, file=None):
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)

    mask = np.zeros_like(workingdata.corr())
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        sns.heatmap(workingdata.corr(), mask=mask, annot=True, cmap='coolwarm')

    plt.title(f'Numeric Attribute Correlations', fontsize=24)

    if file: plt.savefig(file, bbox_inches='tight')
    return

def gen_scatterplots(workingdata, target_column, file=None):
    plotData = workingdata.copy()
    dataColumns = list(plotData.columns)
    for col in dataColumns:
        if is_numeric_dtype(plotData[col]) == False:
            plotData.drop(col, axis=1, inplace=True)
    list_of_columns = [col for col in plotData.columns if col != target_column]

    cols = 2
    rows = math.ceil(len(list_of_columns)/cols)
    figwidth = 10 * cols
    figheight = 6 * rows

    fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (figwidth, figheight))

    color_choices = ['blue', 'red', 'black', 'green', 'purple','crimson']

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    ax = ax.ravel()         # Ravel turns a matrix into a vector... easier to iterate

    for i, column in enumerate(list_of_columns):
        ax[i].scatter(plotData[column], plotData[target_column], color=color_choices[i % len(color_choices)], alpha=0.1)

        ax[i].set_title(f'{column} vs. {target_column}', fontsize=18)

        ax[i].set_ylabel(f'{target_column}', fontsize=14)
        ax[i].set_xlabel(f'{column}', fontsize=14)

    fig.suptitle('\nEach Attribute vs. '+target_column+' - Scatter Plots', size=24)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0, top=0.88)
    if file: plt.savefig(file, bbox_inches='tight')
    plt.show()
    return


def gen_histograms(workingdata, file=None):
    cols=2
    rows = math.ceil(len(workingdata.columns)/cols)
    figwidth = 10 * cols
    figheight = 6 * rows

    fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (figwidth, figheight))

    color_choices = ['blue', 'red', 'black', 'green', 'purple','crimson']
    ax = ax.ravel()         # Ravel turns a matrix into a vector... easier to iterate

    for i, column in enumerate(workingdata.columns):
        ax[i].hist(workingdata[column], color=color_choices[i % len(color_choices)], alpha = 1)

        ax[i].set_title(f'{workingdata[column].name}', fontsize=18)
        ax[i].set_ylabel('Observations', fontsize=18)
        ax[i].set_xlabel('', fontsize=18)

    fig.suptitle('\nHistograms for All Attributes', size=24)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0, top=0.9)
    if file: plt.savefig(file, bbox_inches='tight')
    plt.show()

    return


def gen_boxplots(workingdata, file=None):
    plotData = workingdata.copy()
    dataColumns = list(plotData.columns)
    for col in dataColumns:
        if is_numeric_dtype(plotData[col]) == False:
            plotData.drop(col, axis=1, inplace=True)

    cols=2
    rows = math.ceil(len(plotData.columns)/cols)
    figwidth = 10 * cols
    figheight = 6 * rows

    fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (figwidth, figheight))

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = ax.ravel()

    for i, column in enumerate(plotData.columns):
        ax[i].boxplot(plotData[column])

        ax[i].set_title(f'{plotData[column].name}', fontsize=18)
        ax[i].set_ylabel('', fontsize=14)
        ax[i].set_xlabel('', fontsize=14)
        ax[i].tick_params(labelbottom=False)

    fig.suptitle('\nBoxplots for All Numeric Attributes', size=24)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0, top=0.88)
    if file: plt.savefig(file, bbox_inches='tight')
    plt.show()

    return

#@st.cache(allow_output_mutation=True)
#def generate_av(workingData, target_att):
#    fig = plt.figure()
#    fig.patch.set_facecolor('#E0E0E0')
#    fig.patch.set_alpha(0.7)

#    workingData = AV.AutoViz(filename='', dfte=workingData, depVar=target_att, verbose=2, chart_format='jpg')
#    return workingData


def generate_distPlot(workingData, target_att, analysisPlotsPath):
    cols = workingData.select_dtypes([np.number]).columns
    for col in cols:
        distplot = klib.dist_plot(workingData[col])
        if distplot != None:
            distplot.figure.savefig(analysisPlotsPath+'/Distribution_Plot__'+col+'.png', dpi=100)
    catplot = klib.cat_plot(workingData, figsize=(30, 30))
    if catplot != None:
        catplot.figure.savefig(analysisPlotsPath+'/Categorical_Data.png', dpi=100)
    plt.clf()
    plt.cla()
    return

def drop_numerical_outliers(workingData, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = workingData.select_dtypes(include=[np.number]).apply(lambda x: np.abs(stats.zscore(x)) < z_thresh).all(axis=1)
    # Drop (inplace) values set to be rejected
    workingData.drop(workingData.index[~constrains], inplace=True)

def setup_modelling_data(workingData):
    # Split dataset, 90% for modelling, 10% unseen prediction
    trainingData = workingData.sample(frac=0.9, random_state=42)
    evaluationData = workingData.drop(trainingData.index)
    trainingData.reset_index(drop=True, inplace=True)
    evaluationData.reset_index(drop=True, inplace=True)
    return trainingData, evaluationData

#@st.cache
#def luxOutput(workingData,target_att):
#    luxAllData = workingData.save_as_html(output=True)
#    workingData.intent = [target_att]
#    luxTarget = workingData.save_as_html(output=True)

    #Add code below main to display widgets
    # luxAllData, luxTarget = luxOutput(workingData,target_att)
    # components.html(luxAllData, width=1100, height=350, scrolling=True)
    # components.html(luxTarget, width=1100, height=350, scrolling=True)
#    return luxAllData, luxTarget

def prediction_confusion_matrix(unseen_predictions, target_att):
    cm = confusion_matrix(unseen_predictions[target_att], unseen_predictions['Label'])
    cm_obj = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
    cm_obj.plot()
    cm_obj.ax_.set(
        title='Unseen Data Confusion Matrix',
        xlabel='Predicted',
        ylabel='Actual')
    return plt

def generate_working_directories(projectName):
    analysisPlotsPath = './Projects/' + projectName + '/analysis/plots/'
    analysisReportsPath = './Projects/' + projectName + '/analysis/reports/'
    dataPath = './Projects/' + projectName + '/data/'
    modellingDataPath = './Projects/' + projectName + '/modelling/data/'
    modellingReportsPath = './Projects/' + projectName + '/modelling/reports/'
    modellingModelPath = './Projects/' + projectName + '/modelling/model/'
    modellingAnalysisPath = './Projects/' + projectName + '/modelling/analysis/'
    explainabilityPath = './Projects/' + projectName + '/explainability/'
    create_directory(dataPath)
    create_directory(analysisPlotsPath)
    create_directory(analysisReportsPath)
    create_directory(modellingDataPath)
    create_directory(modellingReportsPath)
    create_directory(modellingModelPath)
    create_directory(modellingAnalysisPath)
    create_directory(explainabilityPath)
    return analysisPlotsPath, analysisReportsPath, modellingDataPath, modellingReportsPath, modellingModelPath, dataPath, modellingAnalysisPath, explainabilityPath

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def csv_to_html(df, th_colour, path, filename):
    style_text = '<style>table {border-collapse: collapse;border-radius: 5px;box-shadow: 0 0 4px rgba(0, 0, 0, 0.25);overflow: hidden;font-family: "Ariel", sans-serif;font-weight: bold;font-size: 14px;}th {background: '+th_colour+';color: #ffffff;text-align: left;}th,td {padding: 10px 20px;}tr:nth-child(even) {background: #eeeeee;}</style>\n'
    out = df.to_html()
    final = style_text + out
    f = open(path+filename, "w")
    f.write(final)
    f.close()

def update_logging (log_list, task, message):
    now = datetime.now()
    timestamp = now.strftime('%d-%m-%Y %H:%M:%S')
    log_list.append([task, message, timestamp])
    return log_list

def generate_classification_model_analysis(model, X_train, y_train, X_test, y_test, modellingAnalysisPath):
    #Generate Confusion Matrix
    try:
        cm = ConfusionMatrix(model)
        cm.fit(X_train, y_train)
        cm.score(X_test, y_test)
        cm.show(outpath="confusion_matrix.png")
        plt.close()
    except Exception as e:
        print(e)
    #Generate Area Under Curve
    try:
        visualizer = ROCAUC(model)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.show(outpath="area_under_curve.png")
        plt.close()
    except Exception as e:
        print(e)
    #Generate classification report
    try:
        visualizer = ClassificationReport(model, support=True)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.show(outpath="classification_report.png")
        plt.close()
    except Exception as e:
        print(e)
    #Generate Precision Recall Curve
    try:
        viz = PrecisionRecallCurve(model)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show(outpath="precison_recall_curve.png")
        plt.close()
    except Exception as e:
        print(e)
    #Generate Learning Curve
    try:
        cv = StratifiedKFold(n_splits=12)
        sizes = np.linspace(0.3, 1.0, 10)
        # Instantiate the classification model and visualizer
        visualizer = LearningCurve(model, cv=cv, scoring='accuracy', train_sizes=sizes, n_jobs=4)
        visualizer.fit(X_train, y_train)  # Fit the data to the visualizer
        visualizer.show(outpath="learning_curve.png")  # Finalize and render the figure
        plt.close()
    except Exception as e:
        print(e)
    #Generate Feature Importances
    try:
        viz = FeatureImportances(model)
        viz.fit(X_train, y_train)
        viz.show(outpath="feature_importance.png")
        plt.close()
    except Exception as e:
        print(e)
    sourcepath = './'
    sourcefiles = os.listdir(sourcepath)
    destinationpath = modellingAnalysisPath
    for file in sourcefiles:
        if file.endswith('.png'):
            shutil.move(os.path.join(sourcepath, file), os.path.join(destinationpath, file))


# noinspection PyPackageRequirements
def generate_regression_model_analysis(model, X_train, y_train, X_test, y_test, modellingAnalysisPath):
    #Generate Residuals plot
    try:
        visualizer = ResidualsPlot(model, hist=True)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.show(outpath="residuals_plot.png")
        plt.close()
    except Exception as e:
        print(e)
    #Generate Prediction Error
    try:
        visualizer = PredictionError(model)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.show(outpath="prediction_error.png")
        plt.close()
    except Exception as e:
        print(e)
    #Generate Learning Curve
    try:
        visualizer = LearningCurve(model, scoring='r2')
        visualizer.fit(X_train, y_train)  # Fit the data to the visualizer
        visualizer.show(outpath="learning_curve")  # Finalize and render the figure
        plt.close()
    except Exception as e:
        print(e)
    #Generate Feature Importances
    try:
        viz = FeatureImportances(model)
        viz.fit(X_train, y_train)
        viz.show(outpath="feature_importance")
        plt.close()
    except Exception as e:
        print(e)
    sourcepath = './'
    sourcefiles = os.listdir(sourcepath)
    destinationpath = modellingAnalysisPath
    for file in sourcefiles:
        if file.endswith('.png'):
            shutil.move(os.path.join(sourcepath, file), os.path.join(destinationpath, file))