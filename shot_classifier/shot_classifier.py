import torch
import torchvision.transforms as transforms
from shot_classifier.ShotClassifierNet import ShotClassifierNet

PATH = './shot_classifier/shot_classifier_net.pth'
classes = ('miss', 'hit', 'kill')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = ShotClassifierNet()
net.eval()
net.to(device)
net.load_state_dict(torch.load(PATH))


transform = transforms.ToTensor()

def classify_shot(img):
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
           # turn the model to evaluate mode
    with torch.no_grad():     # does not calculate gradient
        class_index = net(tensor).argmax()
        prediction = classes[class_index]
        return prediction
