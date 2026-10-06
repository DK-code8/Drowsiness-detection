# Driver Drowsiness Detection

Computer-vision system that detects driver drowsiness from eye state (open/closed) and yawning,
using convolutional neural networks trained on face and eye-crop images.

**Course:** ITC6125 – Machine Learning and Applications, Deree – The American College of Greece (Spring 2025)

## Tech
Python · TensorFlow/Keras (CNNs) · OpenCV · scikit-learn · NumPy · Matplotlib · Pillow

## Contents
- `ML and Apps Project/Drowsiness Detection.ipynb`: main notebook (preprocessing, CNN training, evaluation)
- `ML and Apps Project/Drowsiness Detection (eye_crops run).ipynb`: experiment trained on cropped eye images
- `ML and Apps Project/eye_state_model.keras`, `.../train/*.h5`: trained models
- `ML and Apps Project/Drowsiness_dataset_new/Drowsiness_dataset_new/train/project*.ipynb`: earlier experiments

## Data
The image datasets are **not included**. The notebooks expect a Kaggle-style drowsiness dataset with
`Open`, `Closed`, `yawn` and `no_yawn` classes in `ML and Apps Project/Drowsiness_dataset_new/Drowsiness_dataset_new/{train,test}/`.

## How to run
```bash
pip install tensorflow opencv-python scikit-learn numpy matplotlib pillow jupyter
jupyter notebook "ML and Apps Project/Drowsiness Detection.ipynb"
```
