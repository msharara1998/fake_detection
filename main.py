from tensorflow.keras.models import load_model
import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

class GANClassification:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        
    def preprocess(self, img):
        # Resize to the input shape of the model
        img = img.resize((256, 256))
        # Convert to numpy array and normalize
        img_array = np.array(img)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, img):
        # Open image using PIL
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        # Preprocess the image
        img_array = self.preprocess(img)
        # Make prediction
        result = self.model.predict(img_array)
        print(result)
        return "Real" if result[0][0] > 0.5 else "Fake"

def main():
    gan_classification = GANClassification("models/cnn_95_val_acc.h5")
    # Define the input and output interfaces for Gradio
    input_interface = gr.inputs.Image(shape=(256, 256))
    output_interface = gr.outputs.Textbox(label="Prediction")

    # Create the Gradio interface
    gr.Interface(fn=gan_classification.predict, inputs=input_interface, outputs=output_interface).launch(server_port=8000)

if __name__ == "__main__":
    main()
