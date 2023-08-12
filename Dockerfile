FROM python:3.7-buster 
COPY . /app             
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python3","-m", "flask", "run","--host","0.0.0.0","--port","7860"]