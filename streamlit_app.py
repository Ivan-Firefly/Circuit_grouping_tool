import streamlit as st
import pandas as pd
from main import start

st.title("TCPro to DXF converter")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    dxf_data = main(uploaded_file)  # Get binary DXF data

    if dxf_data:
        st.download_button(
            label="Download DXF",
            data=dxf_data,
            file_name="output.dxf",
            mime="application/dxf"
        )
    else:
        st.error("Error: Failed to generate DXF file.")
