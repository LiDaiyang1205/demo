import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import time

time_start=time.time()

#Hyper Parameters
EPOCH = 1 #training time
BATCH_SIZE = 10
LR = 0.001
DOWNLOAD_MNIST = False

#traindataset
train_data = torchvision.datasets.MNIST(
	root='./mnist',
	train=True,
	transform=torchvision.transforms.ToTensor(),
	download=DOWNLOAD_MNIST
)

#plot
#print(train_data.train_data.size())
#print(train_data.train_labels.size())
#plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
#plt.title('%i'%train_data.train_labels[0])
#plt.show()

#traindata load
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#testdataset
test_data=torchvision.datasets.MNIST( root='./mnist', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10] 
test_y = test_data.test_labels[:10]

#build a cnn
class CNN(nn.Module):
      def __init__(self):
          super(CNN, self).__init__()
          self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
              nn.Conv2d(
                  in_channels=1,              # input height
                  out_channels=16,            # n_filters
                  kernel_size=5,              # filter size
                  stride=1,                   # filter movement/step
                  padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-    1)/2 if stride=1
              ),                              # output shape (16, 28, 28)
              nn.ReLU(),                      # activation
              nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
          )
          self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
              nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
              nn.ReLU(),                      # activation
              nn.MaxPool2d(2),                # output shape (32, 7, 7)
          )
          self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
  
      def forward(self, x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
          output = self.out(x)     
          return output,x                     # return x for visualization

cnn = CNN()
print(cnn) 
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

#Train
for epoch in range(EPOCH):
     for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
 
         output = cnn(b_x)[0]               # cnn output
         loss = loss_func(output, b_y)   # cross entropy loss
         optimizer.zero_grad()           # clear gradients for this training step
         loss.backward()                 # backpropagation, compute gradients
         optimizer.step()                # apply gradients

         if step % 50 == 0:
             test_output, last_layer = cnn(test_x)
             pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
             time_end=time.time()
             print('Step: ', step, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy,'| totally cost',time_end-time_start)

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
