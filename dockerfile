FROM tensorflow/tensorflow:latest-gpu

ENV PYTHONUNBUFFERED 1

RUN pip install -U pip
RUN pip install ipython jupyterlab notebook pandas numpy seaborn scikit-learn

CMD [ "jupyter", "notebook", "--allow-root", "--ip=0.0.0.0" ]
