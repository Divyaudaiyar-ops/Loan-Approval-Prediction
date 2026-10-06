# Loan Approval Prediction

Streamlit application for exploring and predicting loan approval with five
classification algorithms and tuned model variants.

## Run the app on Windows

From PowerShell in this project folder:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If the `venv` folder already exists and its dependencies are installed, just
activate it and start Streamlit:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same PowerShell window,
then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Streamlit prints a local address, usually `http://localhost:8501`. Keep the
terminal open while using the app.

## Train or refresh the saved models

The repository includes saved models under `models/`. To retrain them from the
included dataset, run from the project root:

```powershell
python src\train_models.py
```

Optional tuning and evaluation scripts are in `src/`.
