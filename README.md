# Almost Always Linear Forecasting (AALF)
This code accompanies our journal submission entitled **Almost Always Linear Forecasting (AALF)**.

# Installation 
All experiments were done using Python 3.10.13.

Install requirements manually and skip to "Recreate the experiments"

```
pip install -r requirements.txt
```

Additionally, LaTeX is necessary to recreate all figures from the paper.

# Recreating the experiments

```
python code/train_models.py    # Train all models (this will take a while)
python code/evaluate_models.py # Create prediction files etc.
python code/selection.py       # Fit all classifiers used for selection
python code/run_baselines.py   # Fit all model selection baselines
python code/aal.py             # Run our AALF method
python code/viz.py             # Visualize results
```
