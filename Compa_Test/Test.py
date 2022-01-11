import torch
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 5

torch.manual_seed(1)
base = torch.ones(100, 2)
x0 = torch.normal(2 * base, 1)
y0 = torch.ones(100)
x1 = torch.normal(-2 * base, 1)
y1 = torch.zeros(100)

x = torch.cat((x0, x1), dim=0).type(torch.float32)
y = torch.cat((y0, y1), dim=0).type(torch.float32) #MSELoss()要求为float，如果为int会报错

plt.ion()
plt.show()

net = torch.nn.Sequential(
torch.nn.Linear(2, 10),
torch.nn.ReLU(),
torch.nn.Linear(10, 1)
) #最终只输出一个函数值

optimizer = torch.optim.SGD(net.parameters(), lr=0.1) #学习率为0.2会导致前几次迭代中出现没有预测的效果，虽然最终预测也会准确
loss_func = torch.nn.MSELoss()

for _ in range(20):
    output = net(x) # [200,1]
    output = output.squeeze() #[200,]
    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    prediction = np.int32(output.data.numpy() >= 0.5) #将神经网络输出值大于等于0.5的取为1，小于的取为0
    plt.scatter(x[:, 0], x[:, 1], c=prediction)
    plt.show()
    accuracy = sum(prediction == y.numpy()) / 200
    print(accuracy)

plt.ioff()
plt.show()