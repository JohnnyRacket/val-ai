
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from ShotClassifierDataset import ShotClassifierDataset
from ShotClassifierNet import ShotClassifierNet

# use gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = ShotClassifierDataset(csv_file='shot_classifier_labels.csv', root_dir = 'shot_classifier_imgs', transform = transforms.ToTensor())
print(len(dataset))
in_channel = 1
num_classes = 3
learning_rate = 1e-3
momentum = 0.9
batch_size = 32
num_epochs = 1000


trainset, testset = torch.utils.data.random_split(dataset, [400,228])
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True) 


classes = ('miss', 'hit', 'kill')

killClassifierNet = ShotClassifierNet()
killClassifierNet.to(device)

# loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(killClassifierNet.parameters(), lr=learning_rate, momentum=momentum)

for epoch in range(num_epochs):  # loop over the dataset multiple times
    
    if (epoch % 25 == 0):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for (data, labels) in testloader:
                images = data.to(device=device)
                labels = labels.to(device=device)
                # calculate outputs by running images through the network
                print(images.shape)
                outputs = killClassifierNet(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct / total} %')
        print(epoch)
        trainset, testset = torch.utils.data.random_split(dataset, [400,228])
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True) 
    running_loss = 0.0
    for i, (data, targets) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        data = data.to(device=device)
        targets = targets.to(device=device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = killClassifierNet(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# save net
PATH = './kill_classifier_net.pth'
torch.save(killClassifierNet.state_dict(), PATH)


correct = 0
total = 0
testloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) 

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for (data, labels) in testloader:
        images = data.to(device=device)
        labels = labels.to(device=device)
        # calculate outputs by running images through the network
        outputs = killClassifierNet(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on all images: {100 * correct / total} %')