from pathlib import Path
import gradio as gr
from fastai.learner import load_learner
from fastai.vision.core import PILImage as _PILImage # Use an alias
from PIL import Image as _PILImageModule # Use an alias

# Suppress specific warnings at the beginning if desired
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fastai.learner", message=".*load_learner.*pickle.*")
# The gradio warning will be fixed by using the correct parameter, so no need to suppress it.

# 1. Define PILImageRGB as it was used during the initial training
# This must be defined before load_learner
class PILImageRGB(_PILImage):
    @classmethod
    def create(cls, fn, **kwargs): # fn is a filename/path
        img = _PILImageModule.open(fn)
        img = img.convert("RGBA") # Handle palette transparency
        img = img.convert("RGB")  # Drop alpha
        return cls(img)

# Load the learner
# Ensure 'fruitveg_res50_{version}.pkl' is in the same directory as app.py or provide the correct path
model_path = Path(__file__).parent / 'fruitveg_res50_7final.pkl'
learn = load_learner(model_path)

def classify(image_filepath_from_gradio: str):
    # Converts the string filepath from Gradio to a pathlib.Path object
    img_path = Path(image_filepath_from_gradio)
    
    pred, pred_idx, probs = learn.predict(img_path)
    
    # learn.dls.vocab contains the class names
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(learn.dls.vocab))}

# Gradio interface
iface = gr.Interface(
    fn=classify,
    # Ensures Gradio input provides a filepath string
    inputs=gr.Image(type="filepath", label="Upload Fruit/Vegetable Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Fruit & Vegetable Classifier",
    description="Upload an image of a fruit or vegetable, and the model will predict its type.",
    # Fixes deprecated Gradio parameter
    flagging_mode="never" # Replaces allow_flagging
)

if __name__ == '__main__':
    iface.launch()