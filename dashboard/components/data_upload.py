import streamlit as st
import pandas as pd
from typing import Tuple, Optional
import logging

class DataUploadComponent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def render(self):
        st.subheader("Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Display data statistics
                st.subheader("Data Statistics")
                st.write(data.describe())
                
                return data
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                self.logger.error(f"File upload error: {str(e)}")
                return None