FROM continuumio/miniconda3:23.10.0-1

LABEL maintainer="causal_inference_multiomics"
LABEL description="Causal inference demonstration using TCGA Pan-Cancer Atlas"

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

COPY . .

RUN mkdir -p data/raw results/figures

SHELL ["conda", "run", "-n", "causal_multiomics", "/bin/bash", "-c"]

RUN python -c "import pandas; import numpy; import sklearn; import statsmodels; \
    import networkx; import lifelines; import pingouin; print('All packages OK')"

EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "causal_multiomics", \
     "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
