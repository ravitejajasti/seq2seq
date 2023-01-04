FROM PYTHON:LATEST
WORKDIR /app
COPY ./app
RUN pip install -r requirements.txt
EXPOSE $PORT

COPY requirements.txt /tmp/requirements.txt
CMD gunicorn --workers=4 --bind 0.0.0.0 $PORT app:app