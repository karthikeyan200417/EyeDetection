import os
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

# Load the pre-trained Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image to a fixed size (optional, adjust as needed)
    resized_image = cv2.resize(gray, (64, 64))
    
    # Compute HOG features
    hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, transform_sqrt=True)
    
    return hog_features

# Function to segment eye image and extract features
def segment_and_extract_features(image_path, sample):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert image to grayscale for Haar cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Initialize list to store extracted features
    eye_features = []
    
    # Iterate over detected eyes
    for i, (ex, ey, ew, eh) in enumerate(eyes):
        # Extract the eye region from the image
        eye_region = image[ey:ey+eh, ex:ex+ew]
        
        # Extract HOG features from the eye region
        hog_features = extract_hog_features(eye_region)
        
        # Store the eye region and its features
        eye_features.append((eye_region, hog_features))
    
    return eye_features if len(eye_features) > 0 else None

# Function to predict eye presence using the SVM classifier
def predict_eye(image_path, svm_classifier, pca_transformer):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded properly
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Extract HOG features from the image
    features = segment_and_extract_features(image_path, sample=0)
    
    if not features:
        print("No eyes detected in the image.")
        return
    
    # Extract the HOG features from the detected eye regions
    hog_features = np.array([feat[1] for feat in features])
    
    # Apply PCA transformation
    hog_features_pca = pca_transformer.transform(hog_features)
    
    # Predict using the SVM classifier
    predictions = svm_classifier.predict(hog_features_pca)
    
    # Print and return the prediction results
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            print(f"Eye {i + 1}: Detected as an Eye")
        else:
            print(f"Eye {i + 1}: Detected as Non-Eye")
    
    return predictions

# Load the trained SVM classifier and PCA transformer
svm_model = joblib.load('svm_classifier.pkl')
pca_transformer = joblib.load('pca_transformer.pkl')

# Example usage
example_image_path = 'data/eye/1.jpeg'  # Replace with your image path
predict_eye(example_image_path, svm_model, pca_transformer)
