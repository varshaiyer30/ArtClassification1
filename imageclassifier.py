import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(), 
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='art',transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
					shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='art',transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
					shuffle=True, num_workers=2)

classes = ('Abstract Impressionism', 'Cubism', 'Expressionism', 'Fauvism',
 'Impressionism', 'Post-Impressionism', 'Realism', 'Renaissance', 'Romanticism',
'Surrealism')

class Net(nn.Module):

        def __init__(self):
                super(Net, self).__init__()

                self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16*5*5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))

                x = self.pool(F.relu(self.conv2(x)))

                x = x.view(-1, 16*5*5)

                x = F.relu(self.fc1(x))

                x = F.relu(self.fc2(x))

                x = self.fc3(x)

                return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):

		inputs, labels = data

		inputs, labels = Variable(inputs), Variable(labels)

		optimizer.zero_grad()

		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.data[0]
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' %
				(epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
	images, labels = data
	outputs = net(Variable(images))
	_, predicted = torch.max(outputs.data, 1)
	c = (predicted == labels).squeeze()
	for i in range(4):
		label = labels[i]
		class_correct[label] += c[i]
		class_total[label] += 1

for i in range(10):
	print('Accuracy of %5s : %2d %%' %
		(classes[i], 100*class_correct[i] / class_total[i]))
