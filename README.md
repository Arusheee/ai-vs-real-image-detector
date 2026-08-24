# AI vs Human Image Detector

A Streamlit demo for our trained image classifier — upload an image and get a Real vs AI-Generated prediction with a confidence score.

## Setup

1. Open this folder in VS Code.
2. Make sure `best_modelNew.pt` (our trained model) is in this folder.
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. It opens at `http://localhost:8501`.

## Other scripts

- `evaluate_folder.py` — batch-tests the model on a folder of real/AI images (expects `external-test-images/human` and `external-test-images/ai` subfolders).
- `calibrate_threshold.py` — sweeps decision thresholds to find the one that gives the best accuracy on a labeled test set.

## Model

ResNet18, trained on the Defactify dataset, Tiny-GenImage, and a custom set of images we collected ourselves (ChatGPT, Gemini, and phone camera photos) to improve generalization beyond the original training distribution.