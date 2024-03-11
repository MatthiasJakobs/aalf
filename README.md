# Almost Always Linear Forecasting (AALF)
This code accompanies our ECML-PKDD 2024 submission entitled **Almost Always Linear Forecasting (AALF)**.

# Installation (local)
All experiments were done using Python 3.10.13.

Install requirements manually and skip to "Recreate the experiments"

```
pip install -r requirements.txt
```

Additionally, LaTeX is necessary to recreate all figures from the paper.

# Installation (Docker)
This repo comes with a Dockerfile that you can build to recreate the same environment:

```
docker build -t aalf .
docker run -it --rm \
            -v $(pwd)/code:/aalf/code \
            -v $(pwd)/models:/aalf/models \
            -v $(pwd)/results:/aalf/results \
            -v $(pwd)/plots:/aalf/plots \
            -v $(pwd)/preds:/aalf/preds \
            aalf /bin/bash
```

# Recreating the experiments
First, rerun the main experiments:

```
python code/main.py
```

Afterwards, recreate tables and plots using:

```
python code/cdd_plots.py
python code/evaluation.py
python code/viz.py
```
