import torch
import torchvision.transforms as transforms
from direction_classifier.DirectionClassifierNet import DirectionClassifierNet

PATH = './direction_classifier/direction_classifier_net.pth'
classes = ('nothing', 'left', 'right', 'center')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = DirectionClassifierNet()
net.to(device)
net.load_state_dict(torch.load(PATH))
net.eval()   

transform = transforms.ToTensor()

def classify_direction(img):
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
           # turn the model to evaluate mode
    with torch.no_grad():     # does not calculate gradient
        class_index = net(tensor).argmax()
        prediction = classes[class_index]
        return prediction
