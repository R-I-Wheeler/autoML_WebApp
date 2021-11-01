import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path

import automlbuilder as amb

def app():
    #log_list = st.session_state['log_list']
    #print(st.session_state['log_list'])
    dataPath = st.session_state['dataPath']

    st.markdown('# Project Log')
    df = pd.DataFrame(st.session_state['log_list'], columns=['Main Task', 'Message', 'Timestamp'])
    df = df.drop_duplicates(subset='Message', keep="first")

    amb.csv_to_html(df, '#4B4B4B', dataPath, 'Project_log.html')

    my_file = Path(dataPath + 'Project_log.html')
    HtmlFile = open(my_file, 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    # print(source_code)
    components.html(source_code, height=1000, scrolling=True)
