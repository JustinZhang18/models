FROM tensorflow/tensorflow:2.4.0-gpu
RUN mkdir /models
COPY ./requirements.txt /
ENV PYTHONPATH "$PYTHONPATH:/models"
RUN pip3 install -r /requirements.txt
