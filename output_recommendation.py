import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from fashion_input import prepare_df  # Assuming you have this function

# Function to preprocess the downloaded image(s)
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((64, 64))  # Resize the image to match the input size of your model
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

# Function to predict category using the trained model
def predict_category(model, image_path):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
    prediction = model.predict(preprocessed_image)
    predicted_category_code = np.argmax(prediction)  # Get the category with highest probability
    return predicted_category_code

# Function to retrieve clothing recommendations based on predicted category
def get_recommendations(predicted_category_code, category_names, df):
    # Get the category name based on the predicted category code
    category_name = category_names.get(predicted_category_code, "Unknown")
    if category_name != "Unknown":
        # Filter the DataFrame to get images of the recommended category
        recommended_df = df[df['category_old'] == category_name]
        # Check if recommended_df is empty
        if not recommended_df.empty:
            # Select a random subset of recommendations (you can adjust the number as needed)
            recommendations = recommended_df.sample(n=min(5, len(recommended_df)))['image_path'].tolist()
            return recommendations
        else:
            print("No images found for the recommended category:", category_name)
            return []
    else:
        print("Unknown category code:", predicted_category_code)
        return []


if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model('best_model.keras')  # Change the path if necessary
    
    # Load the mapping of category numbers to category names
    category_names = {
        18: 'Tee',
        6: 'Cardigan',
        17: 'Tank',
        16: 'Sweater',
        3: 'Blouse',
        11: 'Jacket',
        13: 'Parka',
        4: 'Bomber',
        1: 'Anorak',
        0: '002.png'
    }  # Update with your category names
    
    # Load the DataFrame with image paths and category names
    df = prepare_df('data/test_modified2.csv', usecols=['image_path', 'category_old'])
    
    # Assuming you have downloaded an image and want to get recommendations for it
    downloaded_image_path = './test_output.jpg'
    # Predict category of the downloaded image
    predicted_category_code = predict_category(model, downloaded_image_path)
    
    # Get recommendations based on the predicted category
    recommendations = get_recommendations(predicted_category_code, category_names, df)
    
    # Display the recommended clothing images
    print("Recommendations based on predicted category:")
    for recommendation in recommendations:
        img = cv2.imread(recommendation)
        cv2.imshow('Recommended Clothing', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
