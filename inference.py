from pathlib import Path
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder
from LLaMa import RecipeGenerator

class Predictor():
    def __init__(self, weights='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth', token="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"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = blip_decoder(pretrained=weights,
                                  image_size=384, vit='base')
        self.generator = RecipeGenerator(token=token)
        
    def predict_title(self, image):

        im = load_image(image, image_size=384, device=self.device)
        model = self.model
        model.eval()
        model = model.to(self.device)

        with torch.no_grad():
            caption = model.generate(im, sample=True, num_beams=3, max_length=20, min_length=5)
            return caption[0]
        
    def generate_recipe(self, title):
        recipe = self.generator.generate(title)
        return recipe


def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')

    w,h = raw_image.size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./demo.jpg', help='path to image for inference')
    parser.add_argument('--weights', default='./checkpoint2.pth', help='path to weights for inference')
    parser.add_argument('--output_dir', default='./output')  
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    predict = Predictor(weights=args.weights)
    title = predict.predict_title(args.input)
    recipe = predict.generate_recipe(title)
    print(title)
    result_path = os.path.join(args.output_dir, 'result_recipe.txt')

    with open(result_path, 'w') as f:
        f.write(recipe)