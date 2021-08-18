import time
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import weights_init_normal

# import TSN NET
import pdb
from networks.my_TSN import TSN

# VIDEO only network
class DeepVAD_video(nn.Module):

    def __init__(self, args):
        super(DeepVAD_video, self).__init__()

        resnet = models.resnet18(pretrained=True) # set num_ftrs = 512
        #resnet = models.resnet34(pretrained=True) # set num_ftrs = 512

        # self TSN net
        self.tsn = TSN(2, 15, 'RGB', 3, base_model='mobilenetv2')
        
        num_ftrs = 512

        self.lstm_input_size = num_ftrs
        self.lstm_layers = args.lstm_layers
        self.lstm_hidden_size = args.lstm_hidden_size
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        # resnet -1 layer
        #self.features = nn.Sequential(
        #    *list(resnet.children())[:-1]# drop the last FC layer
        #)
        # mob, tsn -1 layer
        self.features = nn.Sequential(
            *list(self.tsn.children())[:-1]# drop the last FC layer
        )

        self.lstm_video = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        self.vad_video = nn.Linear(self.lstm_hidden_size, 2)
        self.dropout = nn.Dropout(p=0.5)

        # reshape video 1000-> 512
        self.video1 = nn.Sequential(
            nn.Conv1d(1000, 512, kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )        
        # reshape audeo 321-> 512
        self.audio1 = nn.Sequential(
           nn.Conv1d(321, 512, kernel_size=5,stride=1,padding=2),
           nn.BatchNorm1d(512),
           nn.ReLU(),
           nn.Linear(50,15),
           nn.ReLU()
        )
        # conv_block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        self.up_sample1 = nn.Conv2d(2,32,kernel_size=1)
        self.max_pool1 = nn.MaxPool2d((1,4))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.up_sample2 = nn.Conv2d(32,64,kernel_size=1)
        self.max_pool2 = nn.MaxPool2d((1,4))

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.up_sample3 = nn.Conv2d(64,128,kernel_size=1)
        self.max_pool3 = nn.MaxPool2d((1,4))

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.up_sample4 = nn.Conv2d(128,256,kernel_size=1)
        self.max_pool4 = nn.MaxPool2d((1,4))

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.up_sample5 = nn.Conv2d(256,512,kernel_size=1)
        self.max_pool5 = nn.MaxPool2d((1,2))


        self.attention = nn.Sequential(
            nn.Linear(4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        self.fully = nn.Sequential(
            nn.Linear(4, 2)
        )


    def weight_init(self, mean=0.0, std=0.02):
        for m in self.named_parameters():
            weights_init_normal(m, mean=mean, std=std)


    def forward(self, x, h, audio):
        #pdb.set_trace()
        # Audio init
        # audio [n_batch,n_frame,n_mels] = [8, 50 ,321]
        B = audio.size(0)
        audio = audio.permute(0,2,1)
        # audio 8, 321, 50
        audio_out = self.audio1(audio)
        audio_out = audio_out.unsqueeze(1)
        # audio 8, 512, 15 -> 8, 1 512 15
        #16, 1, 15, 1, 224, 224
        x = x.squeeze(1)
        # video : [n_batch, n_frame, grayscale, width, height] = [16, 15, 1, 224, 224]
        # 

        batch,frames,channels,height,width = x.size()
        #16, 15, 1, 224, 224
        x = x.view(batch*frames,channels,height,width)
        #240, 1, 224, 224
        x= torch.cat((x,x,x),1)
        #240, 3, 224, 224
        x = self.tsn(x)
        #240, 1000
        x = x.view(batch , -1, frames)
        #16, 1000, 15
        x = self.video1(x)
        #16, 512, 15
        x = x.unsqueeze(1)
        #16, 1, 512, 15

        

        out = x.permute(3,1,0,2)
        out_aud = audio_out.permute(3,1,0,2)

        out = out.squeeze()
        out_aud = out_aud.squeeze()
        if B == 1:
            out = out.unsqueeze(1)
            out_aud = out_aud.unsqueeze(1)
        #15,8,512
        out, _ = self.lstm_video(out, h)
        out_aud, _ = self.lstm_video(out_aud, h)

        out = self.dropout(out[-1])
        out_aud = self.dropout(out_aud[-1])

        out = self.vad_video(out)
        out_aud = self.vad_video(out_aud)

        #out_con = out + 

        out = torch.nn.functional.softmax(out, 1)
        out_aud = torch.nn.functional.softmax(out_aud, 1)

                        
        #pdb.set_trace()
        # attention module 추가
        concat = torch.cat((out, out), 1)
        concat_aud_att = self.attention(concat)
        concat_aud_att = torch.sigmoid(concat_aud_att)

        attaud = concat_aud_att*out
        attvid = (1-concat_aud_att)*out

        concat_attmul = torch.cat((attaud, attvid), 1)
        out_att = self.fully(concat_attmul)
        out_att = torch.sigmoid(out_att)



        #out = torch.unsqueeze(out,0)
        #out_aud = torch.unsqueeze(out_aud,0)
        #concat = torch.cat((out_aud, out), 0)
        #attention = torch.sigmoid(concat)
        return out_att


    def init_hidden(self,is_train):
        if is_train:
            return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                      Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())
