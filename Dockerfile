FROM python:3

COPY . .
RUN pip install flask pandas sklearn nltk pymorphy2 fasttext

CMD ["python3", "main.py"]

