FROM python:3.10

COPY requirements.txt .

RUN pip3 install pipenv
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev

COPY Pipfile .
COPY Pipfile.lock .

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt




COPY . .

ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]