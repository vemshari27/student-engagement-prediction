import os, sys
import numpy
import cv2
import pandas as pd 

import torch
import torch.nn as nn
import network_test
from torch.autograd import Variable
from torchvision import transforms

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train(True)


def load_model():
    base_network = network_test.network_dict["ResNet18"]()
    bottleneck_layer = nn.Linear(base_network.output_num(), 256)
    classifier_layer = nn.Linear(bottleneck_layer.out_features, 2)
    bottleneck_layer = nn.Sequential(bottleneck_layer, nn.ReLU(), nn.Dropout(0.2))
    model = nn.Sequential(base_network, bottleneck_layer, classifier_layer, nn.Softmax())
    model.load_state_dict(torch.load('./weights_new/r18_epoch_13500_updt1.pt'))
    model.eval() 
    model.apply(apply_dropout)
# 
    # model.eval()    
    return model

# def load_video(data):
#     frames = os.listdir(data)

#     x = []
#     for frame in frames:
#         imgpath = os.path.join(data, frame)
#         image = cv2.imread(imgpath)
#         resize_image = cv2.resize(image, (456,256))
#         norm_image = cv2.normalize(resize_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         x.append(numpy.rollaxis(norm_image, 2))
#     vis = numpy.array(x, 'float32') 
#     vis_data = torch.from_numpy(vis)
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#     vis_data = normalize(vis_data)
#     return vis_data
    


def load_video(data):
    videoCapture = cv2.VideoCapture(data)
    x = []
    count = 0
    while(videoCapture.isOpened()):
        retval, image = videoCapture.read()
        
        if retval and count % 25 ==0:
            resize_image = cv2.resize(image, (456,256))
            norm_image = cv2.normalize(resize_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            x.append(numpy.rollaxis(norm_image, 2))
        else:
            break
        count +=1
    vis = numpy.array(x, 'float32')   
    vis_data = torch.from_numpy(vis)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    vis_data = normalize(vis_data)
    return vis_data

    
def predict_trait(data_video, model):
    y = load_video(data_video) 
    print("vid size::", y.size())
    sys.exit(0)
    return model(Variable(y))

if __name__ == "__main__":
    model = load_model()
    df = pd.read_csv('/media/mpl1/mpl_hd2/CHINCHU/idiap_backup/installation/Xlearn/pytorch/data/office/VR/VR182_videodata_testlist.txt', header = None, sep = " ")
    filename = df[0]
    label_ = df[1]
    sum_absolute_errors = 0.0
    start_test = True
    for i in range(0,len(df)):

        vid_data = filename[i]
        y = predict_trait(vid_data, model)
        y = torch.mean(y, dim = 0, keepdim = True)
        print("******Predicted********", y.data.cpu().numpy())
        label = [int(label_[i])]
        # print(label)
        # print(type(y), type(label))
        labels = Variable(torch.FloatTensor(label))
        # print(labels)
        # sys.exit(0)
        if start_test:
            all_predict = y.data.float()
            all_label = labels.data.float()
            start_test = False

        else:
            all_predict = torch.cat((all_predict, y.data.float()), 0)
            all_label = torch.cat((all_label, labels.data.float()), 0)

    _, predict = torch.max(all_predict, dim = 1)
    print("Prediction::",predict)
    print("Actual::",all_label)
    absolute_errors = torch.abs(all_label - predict.float())
    sum_absolute_errors = torch.sum(absolute_errors, 0)
    error = sum_absolute_errors/len(df)
    accuracy = 1-error  

    print('ValueExtraversion, ValueAgreeableness, ValueConscientiousness, ValueNeurotisicm, ValueOpenness')
    print("Validation accuracy of Resnet18 model epoch 13500::", accuracy.tolist())
    print("without drop out")

