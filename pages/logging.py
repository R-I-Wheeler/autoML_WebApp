import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import shutil
import os

import automlbuilder as amb

def app():
    log_list = st.session_state['log_list']

    #print(st.session_state['log_list'])
    dataPath = st.session_state['dataPath']

    st.markdown('# Project Log')
    df = pd.DataFrame(log_list, columns=['Main Task', 'Message', 'Timestamp'])
    #df = df.drop_duplicates(subset=['Message'], keep="first")

    df.to_csv(dataPath + 'Project_log.csv', index=False)
    amb.csv_to_html(df, '#4B4B4B', dataPath, 'Project_log.html')

    my_file = Path(dataPath + 'Project_log.html')
    HtmlFile = open(my_file, 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    # print(source_code)
    components.html(source_code, height=1000, scrolling=True)

    project_name = st.session_state['projectName']

    src_dir = 'HTML_Assets/'
    dest_dir = 'Projects/'+project_name
    files = os.listdir(src_dir)

    # iterating over all the files in
    # the source directory
    for fname in files:
        # copying the files to the
        # destination directory
        shutil.copy2(os.path.join(src_dir, fname), dest_dir)

    shutil.make_archive(project_name, 'zip', './Projects/'+project_name)
    with open(project_name+".zip", "rb") as fp:
        btn = st.download_button(
            label="Download Project",
            data=fp,
            file_name=project_name+".zip",
            mime="application/zip"
        )
    if btn:
        os.remove(project_name+'.zip')


