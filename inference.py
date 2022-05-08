import torch
from torchvision.models import resnet18
import argparse
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize

parser = argparse.ArgumentParser(description='Script to classify checkboxes.')
parser.add_argument('-f', required=True, dest='file_path')
parser.add_argument('--model-path', dest='model_path', default='misc/model_weights/checkbox_resnet18_weights.pt')

if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using Device: ", device)

    model = resnet18()
    model.fc = torch.nn.Linear(512, 3)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    class_list = ['checked', 'other', 'unchecked']

    input_tensor = (ToTensor()(Resize((224,224))(Image.open(args.file_path).convert('RGB'))))
    input_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input_tensor).unsqueeze(0)

    with torch.inference_mode():
        print(class_list[torch.argmax(model(input_tensor)[0])])

