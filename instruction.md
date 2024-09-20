# For Windows
python -m venv venv

# For MacOS/Linux
python3 -m venv venv

# For Windows
venv\Scripts\activate

# For MacOS/Linux
source venv/bin/activate

deactivate

download tesseract-OCR (just google it (not pip install))


utils - downloads
validate - check if image are good
data_prep - cleans - cleaned_train
model - trains saved_models
real_code/sample_code - runs _out
sanity - cheacks format - python .\src\sanity.py --test_filename dataset\test.csv --output_filename dataset\test_out.csv
evaluate - should evaluate