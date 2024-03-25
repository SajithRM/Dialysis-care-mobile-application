

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL_PATH = "../saved_models/1"
MODEL = tf.saved_model.load(MODEL_PATH)

CLASS_NAMES = ['beef_carpaccio', 'beet_salad', 'breakfast_burrito', 'chicken_curry', 'chicken_wings', 'cup_cakes', 'eggs_benedict', 'falafel', 'french_fries', 'fried_rice', 'frozen_yogurt', 'greek_salad', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'ice_cream', 'macaroni_and_cheese', 'omelette', 'onion_rings', 'pizza', 'samosa', 'spring_rolls', 'waffles']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    # Convert the image data to float32 and normalize it to the range [0, 1]
    image = image.astype(np.float32) / 255.0
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Assuming your SavedModel has a 'serving_default' signature
    infer = MODEL.signatures["serving_default"]
    predictions = infer(tf.constant(img_batch))

    # Inspect the keys of the predictions dictionary
    keys = list(predictions.keys())
    print("Available keys in predictions:", keys)

    # Use the first available key as the output key
    output_key = keys[0] if keys else None

    if output_key:
        # Assuming the output key is the first available key
        predicted_class = CLASS_NAMES[np.argmax(predictions[output_key][0])]
        confidence = np.max(predictions[output_key][0])
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'output_key': output_key  # Include the output key in the response
        }
    else:
        return {'error': 'No output key found in predictions'}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8090)
