import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torchsummary import summary
import torch.nn.functional as F
import data.flic_dataset as flic_dataset
import models.DeepNet as deepnet
import models.simplenet as simplenet
import models.residualNet as resnet


def evaluate(model, testloader, device):
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = F.mse_loss(labels, pred)
            loss_sum += loss
        print("loss average: {:.3f}".format(loss_sum / len(testloader)))
    return loss_sum/len(testloader)


def train_model(model,criterion,optimizer,dataload,weight_path,test_loader, device,num_epochs=3):
    test_accuracy_list = []
    epoch_num = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        print("epoch: " + str(epoch_num))
        for images, labels in dataload:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch_num % 5 == 0:
            print("save_weight for epoch " + str(epoch_num))
            torch.save(model.state_dict(), weight_path + str(epoch_num))
        print("test accuracy after epoch" + str(epoch_num) + "\n")
        test_accuracy = evaluate(model, test_loader, device)
        test_accuracy_list.append(test_accuracy.cpu().numpy())
        epoch_num += 1
        print(f"Training loss: {epoch_loss/len(dataload)}")
    return test_accuracy_list


if __name__ == '__main__':
    model_select = "mpii"
    if model_select == "deepnet":
        model = deepnet.DeepNet(9)
        weight_path = "./deepnet_weight/weight_epoch"
        lr = 0.00001
    elif model_select == "resnet":
        model = resnet.ResidualNet(9)
        weight_path = "./residual_weight/weight_epoch"
        lr = 0.0001
    elif model_select == "simplenet":
        model = simplenet.simpleNet(9)
        weight_path = "./simple_weight/weight_epoch"
        lr = 0.0001
    elif model_select == "mpii":
        model = resnet.ResidualNet(16)
        weight_path = "./mpii_weight/weight_epoch"
        lr = 0.00001
    print("init model")
    # model = model.ResidualNet(9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = model.to(device)
    summary(model, input_size=(3, 220, 220))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = nn.modules.loss.MSELoss()
    print("Load dataset")
    trainset = flic_dataset.mydataset("./data/FLIC/train_joints.csv", "./data/FLIC/images")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= 10, shuffle=True)
    testset = flic_dataset.mydataset("./data/FLIC/test_joints.csv", "./data/FLIC/images")
    testloader = torch.utils.data.DataLoader(testset)
    print("start training")
    test_list = train_model(model, loss, optimizer, trainloader, weight_path, testloader, device, 101)
    print(test_list)







