import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.allmodels import build_densenet121, build_resnet50, build_efficientnetb0, build_efficientnetb4, build_efficientnetb5, build_efficientnetb6, build_efficientnetb7
from skimage.transform import resize
from skimage.io import imread
from skimage.color import gray2rgb

# Dataset directory and model options
DATASET_DIR = './datasets'
MODEL_OPTIONS = {
    '1': 'DenseNet121',
    '2': 'ResNet50',
    '3': 'EfficientNetB0',
    '4': 'EfficientNetB4',
    '5': 'EfficientNetB5',
    '6': 'EfficientNetB6',
    '7': 'EfficientNetB7'
}

# Helper function to resize images
def resize_images(df, directory, target_size):
    resized_images = []
    for filename in df['filename']:
        img_path = os.path.join(directory, filename)
        img = imread(img_path)
        if img.ndim == 2:  # Grayscale image
            img = gray2rgb(img)
        img_resized = resize(img, target_size, anti_aliasing=True)
        resized_images.append(img_resized)
    return np.array(resized_images)

# Main script
def main():
    print("Welcome to the Image Training Program!")
    
    # Step 1: Select Dataset
    print("\nAvailable datasets:")
    datasets = os.listdir(DATASET_DIR)
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset}")
    
    dataset_choice = input("\nSelect the dataset by number: ")
    dataset_path = os.path.join(DATASET_DIR, datasets[int(dataset_choice)-1])
    
    # Load the dataset (assuming a CSV with 'filename' and 'count' columns)
    print(f"Loading dataset: {datasets[int(dataset_choice)-1]}")
    df = pd.read_csv(dataset_path)
    
    # Normalize the target column (assuming 'count')
    mean_count = np.mean(df['count'])
    std_count = np.std(df['count'])
    df['count_normalized'] = (df['count'] - mean_count) / std_count
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Step 2: Select Model
    print("\nAvailable models:")
    for key, model_name in MODEL_OPTIONS.items():
        print(f"{key}. {model_name}")
    
    model_choice = input("\nSelect the model by number: ")
    
    # Step 3: Load the chosen model
    input_shape = (224, 224, 3)
    if model_choice == '1':
        model = build_densenet121(input_shape)
    elif model_choice == '2':
        model = build_resnet50(input_shape)
    elif model_choice == '3':
        model = build_efficientnetb0(input_shape)
    elif model_choice == '4':
        model = build_efficientnetb4(input_shape)
    elif model_choice == '5':
        model = build_efficientnetb5(input_shape)
    elif model_choice == '6':
        model = build_efficientnetb6(input_shape)
    elif model_choice == '7':
        model = build_efficientnetb7(input_shape)
    else:
        print("Invalid choice, exiting.")
        return
    
    print(f"Selected Model: {MODEL_OPTIONS[model_choice]}")
    
    # Step 4: Prepare the data
    target_size = (224, 224)
    X_train = resize_images(train_df, './datasets/images', target_size)
    X_test = resize_images(test_df, './datasets/images', target_size)
    y_train = train_df['count_normalized'].values
    y_test = test_df['count_normalized'].values

    # Step 5: Setup data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=16)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=16, shuffle=False)

    # Step 6: Train the model
    print("Training the model...")
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )
    
    # Step 7: Evaluate the model
    print("Evaluating the model...")
    scores = model.evaluate(test_generator)
    print(f"Test Loss: {scores[0]}, Test MAE: {scores[1]}")
    
    # Step 8: Get Predictions
    print("Getting predictions on the test set...")
    y_pred = model.predict(test_generator)
    y_pred = y_pred.flatten()  # Flatten predictions
    
    # Step 9: De-normalize the predictions and ground truth values
    y_pred_denormalized = (y_pred * std_count) + mean_count
    y_test_denormalized = (y_test * std_count) + mean_count
    
    # Step 10: Calculate MAE and MSE
    mae = mean_absolute_error(y_test_denormalized, y_pred_denormalized)
    mse = mean_squared_error(y_test_denormalized, y_pred_denormalized)
    
    print(f"\nMean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

if __name__ == "__main__":
    main()
