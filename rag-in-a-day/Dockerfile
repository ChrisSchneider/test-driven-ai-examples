FROM registry.access.redhat.com/ubi9/python-311

RUN pip install poetry
ADD --chown=1001:0 pyproject.toml pyproject.toml
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --only main

ADD --chown=1001:0 . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
