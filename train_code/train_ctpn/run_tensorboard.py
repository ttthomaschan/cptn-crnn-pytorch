import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

device = "cuda:3" if torch.cuda.is_available() else "cpu"
print(device)

# 设置输出文件目录
writer = SummaryWriter('runs/test')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))
print(len(images))
print(images.device)
print(next(model.parameters()).device)
images = images.to(device)
model.to(device)
print(images.device)
print(next(model.parameters()).device)
grid = torchvision.utils.make_grid(images)
writer.add_images('images', images)
# 写入图像数据 
writer.add_graph(model, images)
# 写入模型
writer.close()