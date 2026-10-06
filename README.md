# Driver Drowsiness Detection

Computer-vision system that detects driver drowsiness from eye state (open/closed) and yawning,
using convolutional neural networks trained on face, eye-crop and mouth images.

**Course:** ITC6125 – Machine Learning and Applications, Deree – The American College of Greece (Spring 2025)

## Tech
Python · TensorFlow/Keras (CNNs) · OpenCV · scikit-learn · NumPy · Matplotlib · Pillow

## Contents
- `ML and Apps Project/Drowsiness Detection.ipynb`: main notebook (preprocessing, CNN training, evaluation)
- `ML and Apps Project/Drowsiness Detection (full run).ipynb`: full end-to-end run
- `ML and Apps Project/Drowsiness Detection (eye_crops run).ipynb`: experiment trained on cropped eye images
- `ML and Apps Project/eye_state_model.keras`: trained eye-state (open/closed) CNN
- `ML and Apps Project/yawn_mouth_simple_cnn2.keras`: trained yawn-detection CNN
- `ML and Apps Project/Drowsiness_dataset_new/Drowsiness_dataset_new/train/`: earlier experiments (`project*.ipynb`, `.h5` models)

## Data
The image datasets are **not included**. The notebooks expect a Kaggle-style drowsiness dataset with
`Open`, `Closed`, `yawn` and `no_yawn` classes in `ML and Apps Project/Drowsiness_dataset_new/Drowsiness_dataset_new/{train,test}/`.

## How to run
```bash
pip install tensorflow opencv-python scikit-learn numpy matplotlib pillow jupyter
jupyter notebook "ML and Apps Project/Drowsiness Detection.ipynb"
```
