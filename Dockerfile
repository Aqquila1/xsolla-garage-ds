FROM python:3

COPY . .
RUN pip install flask pandas sklearn nltk pymorphy2

CMD ["python3", "main.py"]

