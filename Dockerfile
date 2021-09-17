FROM python:3.7
EXPOSE 8501
WORKDIR /app
RUN apt-get update
RUN apt-get install libgl1
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD streamlit run app.py