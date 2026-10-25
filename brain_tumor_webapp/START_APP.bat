@echo off
TITLE NeuroVision AI - Brain MRI Classifier
call conda activate brain_ssl
if errorlevel 1 (
  echo Could not activate the brain_ssl environment.
  echo Open Anaconda Prompt and run: conda activate brain_ssl
  pause
  exit /b 1
)
python -m streamlit run app.py
pause
