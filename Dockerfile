FROM tensorflow/tensorflow

EXPOSE 8000

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]