FROM python:3.8.18

WORKDIR /workspace

COPY requirements.txt .
COPY streamlit_demo.py .
COPY common_functions.py .

RUN pip install --upgrade pip
RUN pip install streamlit
RUN pip install -r /workspace/requirements.txt
EXPOSE 7860
ENTRYPOINT ["streamlit", "run", "streamlit_demo.py", "--server.port=8501", "--server.address=0.0.0.0"]
