FROM tensorflow/tensorflow:latest-gpu

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN pip install -U pip
RUN pip install ipython jupyterlab notebook pandas numpy seaborn scikit-learn scikit-image tqdm

CMD [ "jupyter", "notebook", "--allow-root", "--ip=0.0.0.0" ]
