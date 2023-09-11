# !pip install -q ultralytics gradio
# from google.colab import files

from io import BytesIO
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# DRIVE_ROOT_PATH = "/content/drive/MyDrive/Colab Notebooks"
DRIVE_ROOT_PATH = "D:\OneDrive\OneDrive - Mastek Limited\My_PC\Projects\Demos_and_Poc\FunkoDemo\Data"

# List of background image paths
background_image_paths = [
    # "/ImagePlaceholding/BestDummy.png"
    "\AdobeColorFunko\Outfits\DummyDress1.png",
    "\AdobeColorFunko\Outfits\GlassesDummy.png",
    "\AdobeColorFunko\Outfits\DummyDress3.png"
]


### For Beard Style
class BeardClassifier:
    def __init__(self, model_path, class_names):
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)  # Load model based on CUDA availability
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image

    def load_model(self, model_path):
        if torch.cuda.is_available():
            # Load model on CUDA if available
            self.model.load_state_dict(torch.load(model_path))
        else:
            # Load model on CPU
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_beard(self, image_path):
        input_image = self.preprocess_image(image_path)

        with torch.no_grad():
            predictions = self.model(input_image)

        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]

        return predicted_label

# For Beard Color
class BeardColorClassifier:
    def __init__(self, model_path, class_names):
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)  # Load model based on CUDA availability
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image

    def load_model(self, model_path):
        if torch.cuda.is_available():
            # Load model on CUDA if available
            self.model.load_state_dict(torch.load(model_path))
        else:
            # Load model on CPU
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_beard_color(self, image_path):
        input_image = self.preprocess_image(image_path)

        with torch.no_grad():
            predictions = self.model(input_image)

        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]

        return predicted_label



# ===================for beard style==========================
def predict_beard_style(image_path):

    # Provide the path to your trained model and the list of class names
    model_path = DRIVE_ROOT_PATH + '/FunkoSavedModels/FunkoResnet18Style.pt'
    class_names = ['Bandholz', 'FullGoatee', 'Moustache', 'RapIndustryStandards', 'ShortBeard']

    beard_classifier = BeardClassifier(model_path, class_names)

    input_image = beard_classifier.preprocess_image(image_path)  # Corrected line

    with torch.no_grad():
        predictions = beard_classifier.model(input_image)  # Use beard_classifier.model here

    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    predicted_style_label = beard_classifier.class_names[predicted_class]  # Use beard_classifier.class_names here

    print(f"The predicted beard style is: {predicted_style_label}")

    return predicted_style_label

# ========================================================================


# ===================for beard color==========================
def predict_beard_color(image_path):

    # Provide the path to your trained model and the list of class names
    color_model_path = DRIVE_ROOT_PATH + '/FunkoSavedModels/FunkoResnet18Color.pt'  # Replace with the actual path to your model
    class_names = ['Black', 'DarkBrown', 'Ginger', 'LightBrown', 'SaltAndPepper', 'White']

    beard_color_classifier = BeardColorClassifier(color_model_path, class_names)

    input_image = beard_color_classifier.preprocess_image(image_path)

    with torch.no_grad():
        predictions = beard_color_classifier.model(input_image)

    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    predicted_color_label = beard_color_classifier.class_names[predicted_class]

    print(f"The predicted beard color is: {predicted_color_label}")

    return predicted_color_label

# ========================================================================


# to set dummy eyes
def dummy_eye(background_image, x,y, placeholder_image_path, x_coordinate, y_coordinate):
    placeholder_image = Image.open(placeholder_image_path)
    target_size = (x, y)
    placeholder_image = placeholder_image.resize(target_size, Image.LANCZOS)
    # placeholder_array = np.array(placeholder_image)
    placeholder_width, placeholder_height = placeholder_image.size
    region_box = (x_coordinate, y_coordinate, x_coordinate + placeholder_width, y_coordinate + placeholder_height)
    placeholder_mask = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None
    background_image.paste(placeholder_image, region_box, mask=placeholder_mask)
    # background_array = np.array(background_image)
    # placeholder_alpha = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None
    #display(background_image)

    return background_image

# funtion which process and set's the beard on the dummy
def process_image_Beard(background_image, x, placeholder_image_path, x_coordinate, y_coordinate):
    placeholder_image = Image.open(placeholder_image_path)
    target_size = (x, x)
    placeholder_image = placeholder_image.resize(target_size, Image.LANCZOS)
    # placeholder_array = np.array(placeholder_image)
    placeholder_width, placeholder_height = placeholder_image.size
    region_box = (x_coordinate, y_coordinate, x_coordinate + placeholder_width, y_coordinate + placeholder_height)
    placeholder_mask = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None
    background_image.paste(placeholder_image, region_box, mask=placeholder_mask)
    background_array = np.array(background_image)
    # placeholder_alpha = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None
    # display(background_image)
    # Convert the resulting image to base64
    # buffered = BytesIO()
    # background_image.save(buffered, format="PNG")
    # base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # print(base64_image)
    
    return background_array


def getDummyBackgroundImage(background_image):
    # dummy with a suite = BestDummy
    background_image = Image.open(DRIVE_ROOT_PATH + background_image)
    # dummy eyebrow
    placeholder_image_eyebro = Image.open(DRIVE_ROOT_PATH + '/AdobeColorFunko/EyezBrowz/Eyebrow.png')
    placeholder_image_eyebro = placeholder_image_eyebro.resize((200,200),Image.LANCZOS)
    # placeholder_array_eyebro = np.array(placeholder_image_eyebro)
    # Define the coordinates of the region to paste the placeholder image
    x_coordinate = 115
    y_coordinate = 80
    # Get the width and height of the placeholder image
    placeholder_width, placeholder_height = placeholder_image_eyebro.size
    # Define the region box as a tuple (x1, y1, x2, y2)
    region_box = (x_coordinate, y_coordinate, x_coordinate + placeholder_width, y_coordinate + placeholder_height)
    placeholder_mask = placeholder_image_eyebro.split()[3] if placeholder_image_eyebro.mode == 'RGBA' else None
    # Paste the placeholder image onto the background image
    background_image.paste(placeholder_image_eyebro, region_box, mask=placeholder_mask)
    # background_array = np.array(background_image)

    return background_image


def setEyesOnTheDummy(background_image):
    # Genders = ['Male','Female']
    predicted_gender = 'Male'
    image_with_eyes = None

    # First function call
    if predicted_gender == 'Male':
        x=245
        y=345
        placeholder_image_path = DRIVE_ROOT_PATH + f'/AdobeColorFunko/EyezBrowz/{predicted_gender}Eye.png'
        x_coordinate = 90
        y_coordinate = 50

        print("++++",type(background_image))
        image_with_eyes = dummy_eye(background_image,x,y, placeholder_image_path, x_coordinate, y_coordinate)

    return image_with_eyes

def setGlassesonDummy(background_image):
    #for glasses
    placeholder_image_glasses = Image.open(DRIVE_ROOT_PATH + '/AdobeColorFunko/Glasses/Glasses.png')
    # placeholder_image_glasses = Image.open("/content/drive/MyDrive/AdobeColorFunko/Glasses/Glasses.png")
    placeholder_image_glasses = placeholder_image_glasses.resize((280,380),Image.LANCZOS)
    placeholder_array_glasses = np.array(placeholder_image_glasses)
    # Define the coordinates of the region to paste the placeholder image
    x_coordinate = 72
    y_coordinate = 30
    # Get the width and height of the placeholder image
    placeholder_width, placeholder_height = placeholder_image_glasses.size
    # Define the region box as a tuple (x1, y1, x2, y2)
    region_box = (x_coordinate, y_coordinate, x_coordinate + placeholder_width, y_coordinate + placeholder_height)
    placeholder_mask = placeholder_image_glasses.split()[3] if placeholder_image_glasses.mode == 'RGBA' else None

    print(">>>>>",type(background_image))
    background_image.paste(placeholder_image_glasses, region_box, mask=placeholder_mask)
    background_array = np.array(background_image)

    return background_image

def generatePopFigure(background_image, predicted_style_label, predicted_color_label):
   
    match predicted_style_label:
        case "Bandholz":
            x=320
            placeholder_image_path = DRIVE_ROOT_PATH + f'/AdobeColorFunko/Bandholz/{predicted_color_label}.png'
            x_coordinate = 50
            y_coordinate = 132

        case "ShortBeard":
            x=300
            placeholder_image_path = DRIVE_ROOT_PATH + f'/AdobeColorFunko/ShortBeard/{predicted_color_label}.png'
            x_coordinate = 62
            y_coordinate = 118

        case "FullGoatee":
            x=230
            placeholder_image_path = DRIVE_ROOT_PATH + f'/AdobeColorFunko/Goatee/{predicted_color_label}.png'
            x_coordinate = 96
            y_coordinate = 162
        
        case "RapIndustryStandards":
            x=290
            placeholder_image_path = DRIVE_ROOT_PATH + f'/AdobeColorFunko/RapIndustry/{predicted_color_label}.png'
            x_coordinate = 67
            y_coordinate = 120

        case "Moustache":
            x=220
            placeholder_image_path = DRIVE_ROOT_PATH + f'/AdobeColorFunko/Moustache/{predicted_color_label}.png'
            x_coordinate = 100
            y_coordinate = 160
        case _:
            print("Sorry, I still don't know how to recognize this!")

    final_pop_image = process_image_Beard(background_image.copy(),x, placeholder_image_path, x_coordinate, y_coordinate)

    return final_pop_image


def getFunkoPOPFigure(image):

    print("Input Image: ", image)
    beard_style = predict_beard_style(image)
    beard_color = predict_beard_color(image)

    final_img_list = []

    # fetch base funko dummy with a suite & eyebrows
    for dummy in background_image_paths:
        print("bg_img_path: ", dummy)
        funkoPopDummy = getDummyBackgroundImage(dummy)

        # set eyes on the dummy
        funkoPopDummyWithEyes = setEyesOnTheDummy(funkoPopDummy)

        #set glassed on the dummy
        # funkoPopDummyWithGlasses = setGlassesonDummy(funkoPopDummyWithEyes)

        # get a POP Figure
        pop_image = generatePopFigure(funkoPopDummyWithEyes, beard_style, beard_color)
        final_img_list.append(pop_image)

    # set glasses on 1st the dummy
    # final_img_list[0] = setGlassesonDummy(final_img_list[0])

    print("final result: ", final_img_list)
    return final_img_list


if __name__ == "__main__":

    theme = gr.themes.Base().set(
        body_background_fill="linear-gradient(180deg,#0e5c99,#017cec)",
        body_background_fill_dark="linear-gradient(180deg,#0e5c99,#017cec)",
        body_text_color="white",
        body_text_color_dark="white",
        button_primary_background_fill="#fbc051",
        button_primary_background_fill_dark="#fbc051",
        button_primary_text_color="black",
        button_primary_text_color_dark="black"
        )

    imageComponent = gr.Image(type="filepath")
    with gr.Blocks(theme=theme, title="POP! Yourself", css="footer {visibility: hidden}") as demo:

        gr.Markdown("""
    <img src="https://funko.com/on/demandware.static/Sites-FunkoUS-Site/-/en_US/v1693988598802/lib/img/pop-yourself-logo.7f7a42c2.svg" width="80" alt="logo">

    ### Get your Funko Pop Today. Get started with our Pop Figure generator tool & generate your Funko avatar quickly by just uploading your image.
    ---""")
        gr.Interface(
            fn=getFunkoPOPFigure,
            inputs=imageComponent,
            outputs=["image", "image", "image"],
            title="The new AI powered POP! Generator",
            description="Upload a clear image with clear background which shows your face completely. The image should not be blurry.",
            allow_flagging="never",
            # examples=[DRIVE_ROOT_PATH+i for i in background_image_paths]
        )

    demo.launch()