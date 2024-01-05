# Sample RAG application (Work in Progress)

A sample smart assistant app to demonstrate how to quickly build a high quality Generative AI app with Test-Driven Development.

[Learn more](https://medium.com/test-driven-ai)


### Uses

* Streamlit
* Llama2 on IBM watsonx
* Elastic Search
* E5 Multilingual Embeddings
* Pytest
* Playwright


### Prereqs

* Local development environment, e.g. [VSCode with Python](https://code.visualstudio.com/docs/python/python-tutorial)
* [Poetry package manager](https://python-poetry.org/docs/)
* [IBM watsonx.ai project and API Key](https://medium.com/the-power-of-ai/ibm-watsonx-ai-the-interface-and-api-e8e1c7227358)
* Elastic Search


### Get started

Set environment variables

    WATSONX_URL=https://us-south.ml.cloud.ibm.com
    WATSONX_API_KEY=xxx
    WATSONX_PROJECT_ID=xxx

Install dependencies

    poetry install

Run unit tests

    pytest

Run app

    streamlit run app.py


### Container commands

Build container:

    docker build -t rag-in-a-day .

Run container:

    docker run --it -p 8501:8501 --env-file .env rag-in-a-day
