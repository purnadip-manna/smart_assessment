FROM python:3.9.9
ADD requirements.txt /requirements.txt 
ADD app.py /app.py
ADD prediction.py /prediction.py
ADD BertDataGenerator.py /BertDataGenerator.py
ADD okteto-stack.yaml /okteto-stack.yaml
RUN pip install -r requirements.txt
EXPOSE 8080
COPY ./static static
CMD ["python","app.py"]