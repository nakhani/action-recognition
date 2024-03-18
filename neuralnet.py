from random import shuffle
from torch import nn
import torch , pickle , glob , os
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader



class actiondata():
    def __init__(self ,  kind , data_path = "./dataset/lists/"):
        self.data_path = data_path
        try:
            if kind == "train":
                with open("./dataset/final_dataset/train.pkl","rb") as f:
                    self.data = pickle.load(f)
            if kind == "validation":
                with open("./dataset/final_dataset/val.pkl","rb") as f:
                    self.data = pickle.load(f)
        except:
            print("split data into train and validation !")

    def train_val(self):
        pickles = glob.glob(os.path.join(self.data_path,"*"))
        labels = []
        data = []
        for pic in pickles:
            label = os.path.split(pic)[1].split(".")[0]
            labels.append(label)
            with open(pic , "rb") as f:
                lst = pickle.load(f)
            for item in lst :
                data.append((label,item))

        devision = int(0.7 * len(data))
        with open("./dataset/final_dataset/labels.pkl","wb") as f:
            pickle.dump(labels,f)

        shuffle(data)
       

        with open("./dataset/final_dataset/train.pkl","wb") as f:
            pickle.dump(data[:devision],f)
        with open("./dataset/final_dataset/val.pkl","wb") as f:
            pickle.dump(data[devision:],f)

    def __getitem__(self,index):
        with open("./dataset/final_dataset/labels.pkl","rb") as f:
            labels = pickle.load(f)
        return transforms.ToTensor()(self.data[index][1]).float() , torch.LongTensor([labels.index(self.data[index][0])])
  
    def __len__(self):
        return len(self.data)


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.l1 = nn.Linear(50,30)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(30,10)
        self.l3 = nn.Linear(10,5)
        self.l4 = nn.Linear(15,5)
        self.l5 = nn.Linear(5,2)

    def forward(self,x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.relu(self.l3(x2))
        x4 = nn.Flatten(x3)
        x5 = self.relu(self.l4(x4))
        x6 = self.relu(self.l4(x5))
        return x6


if __name__ == "__main__":

    train_dataset = actiondata("train") 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size = 10,
                                            shuffle = True,
                                            pin_memory=True)

    val_dataset = actiondata("validation") 
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size = 10,
                                            shuffle = True,
                                            pin_memory=True)

    Model = classifier()

    CUDA = torch.cuda.is_available()

    if CUDA:
        Model = Model.cuda()
        
    Loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(),lr = 0.000001)



    lrr = 0.001
    count = 1
    epoch = 7500
    desiredacc = 82
    iter = 0
    for i in range(epoch):
        for Images,targs in train_loader:
            iter += 1
            if CUDA:
                Images = Variable(Images.cuda())
                targs = Variable(targs.cuda())
            else:
                Images = Variable(Images)
                targs = Variable(targs)
                
            optimizer.zero_grad()
            outputs = Model(Images)
            loss = Loss_fn(outputs,targs.squeeze(1))
            loss.backward()
            optimizer.step()
            
            if (iter+1)%150 == 0:
                correct = 0
                total = 0
                for images,labels in val_loader:
                    if CUDA:
                        images = Variable(images.cuda())
                    else:
                        images = Variable(images)
                    outputs = Model(images)
                    _,predicted = torch.max(outputs.data,1)
                    labels = labels.squeeze(1)
                    total += labels.size(0)
                    if CUDA:
                        correct += (predicted.cpu()==labels.cpu()).sum()
                    else:
                        correct += (predicted==labels).sum()
                accuracy = 100 * correct / total
                print('after {} epochs '.format(i) ,'accuracy is {} % , Loss is {}'.format(accuracy,loss))
                if accuracy >= desiredacc:
                    desiredacc += 1
                    lrr = lrr / 3
                    print('SGD with Learning rate {} \n'.format(lrr))
                    optimizer = torch.optim.SGD(Model.parameters(),lr = lrr)  
                if accuracy >= 85:
                    torch.save(Model.state_dict(),'./gdrive/My Drive/action_recognition/result_{}_{}_{}.pt'.format(i,accuracy,count))
                    count += 1