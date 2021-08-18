from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from utils import utils as utils
from torch.utils.data import DataLoader
import time
import torch.nn.utils as torchutils
from torch.autograd import Variable
from utils.logger import Logger
import os
import sys
from dcase_util.data import ProbabilityEncoder
import torch.nn.functional as F
import pdb



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
    parser.add_argument('--time_depth', type=int, default=15, help='number of time frames in each video\audio sample')
    parser.add_argument('--workers', type=int, default=0, help='num workers for data loading')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum factor')
    parser.add_argument('--save_freq', type=int, default=1, help='freq of saving the model')
    parser.add_argument('--print_freq', type=int, default=50, help='freq of printing stats')
    parser.add_argument('--seed', type=int, default=44974274, help='random seed')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of lstm layers in the model')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='number of neurons in each lstm layer in the model')
    parser.add_argument('--use_mcb', action='store_true', help='wether to use MCB or concat')
    parser.add_argument('--mcb_output_size', type=int, default=1024, help='the size of the MCB outputl')
    parser.add_argument('--debug', action='store_true', help='print debug outputs')
    parser.add_argument('--freeze_layers', action='store_true', help='wether to freeze the first layers of the model')
    parser.add_argument('--arch', type=str, default='AV', choices=['Audio', 'Video', 'AV'], help='which modality to train - Video\Audio\Multimodal')
    parser.add_argument('--pre_train', type=str, default='/home/nas2/user/jsydshs/VVAD_GIT/MM_VAD/saved_models/Video/acc_90.447_epoch_003_arch_Video.pkl', help='path to a pre-trained network')

    args = parser.parse_args()
    print(args, end='\n\n')

    torch.manual_seed(args.seed)
    
    # focal loss
    class FocalLoss(nn.Module):
        def __init__(self, gamma=0, alpha=None, size_average=True):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha
            if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
            if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
            self.size_average = size_average

        def forward(self, input, target):
            if input.dim()>2:
                input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1,1)

            logpt = F.log_softmax(input)
            logpt = logpt.gather(1,target)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            if self.alpha is not None:
                if self.alpha.type()!=input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0,target.data.view(-1))
                logpt = logpt * Variable(at)

            loss = -1 * (1-pt)**self.gamma * logpt
            if self.size_average: return loss.mean()
            else:return loss.sum()



    # set the logger
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = Logger('logs')

    # create a saved models folder
    save_dir = os.path.join('saved_models', args.arch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create train + val datasets
    dataset = utils.import_dataset(args)

    train_dataset = dataset(DataDir='/data/MOBIO/MOBIO_FULL_train_refined/',audio_DataDir = "/data/MOBIO/MOBIO_FULL_train_refined_WAVF/", timeDepth = args.time_depth, is_train=True)
    val_dataset = dataset(DataDir='/data/MOBIO/MOBIO_FULL_val_refined/',audio_DataDir = "/data/MOBIO/MOBIO_FULL_val_refined_WAVF/", timeDepth = args.time_depth, is_train=False)
    
    #train_dataset = dataset(DataDir='/home/nas2/user/jsydshs/VVAD_GIT/data/MOBIO/testing_v/',audio_DataDir = "/home/nas2/user/jsydshs/VVAD_GIT/data/MOBIO/testing_a/", timeDepth = args.time_depth, is_train=True)
    #val_dataset = dataset(DataDir='/home/nas2/user/jsydshs/VVAD_GIT/data/MOBIO/testing_v/',audio_DataDir = "/home/nas2/user/jsydshs/VVAD_GIT/data/MOBIO/testing_a/", timeDepth = args.time_depth, is_train=False)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    # create the data loaders
    
    check1=time.time()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
        drop_last=True)
#    print("check2: {}".format(time.time()-check1))
    check2=time.time()
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        drop_last=True)
#    print("check3: {}".format(time.time()-check2))
    check3=time.time()
    # import network
    net = utils.import_network(args)

    # create optimizer and loss (optionaly assign each class with different weight
    weight = torch.FloatTensor(2)
    weight[0] = 1  # class 0 - non-speech
    weight[1] = 1  # class 1 - speech
    criterion = nn.CrossEntropyLoss(weight=weight).cuda()
    # create optimizer and loss
    #criterion = FocalLoss().cuda()
    

    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14, 20, 100], gamma=0.5) #14,20,100 epoch때 lr을 절반으로 줄인다. 
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor =0.1, patience =10 , min_lr =0) 
    # init from a saved checkpoint

    if args.pre_train is not '':
        model_name = os.path.join('pre_trained', args.arch, args.pre_train)

        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            pretrained = checkpoint['state_dict']
            net.load_state_dict(pretrained,strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_name, checkpoint['epoch']))
        else:
            print('Couldn\'t load model from {}'.format(model_name))
    else:
        print('Training the model from scratch.')

    # freeze layeres

    def freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    if args.arch == 'Video' and args.freeze_layers == True:  # Multimodal 
        freeze_layer(net.features)

    if args.arch == 'Audio' and args.freeze_layers == True:
        freeze_layer(net.wavenet_en)
        freeze_layer(net.bn)

    if args.arch == 'AV' and args.freeze_layers == True:
        freeze_layer(net.features)
        freeze_layer(net.wavenet_en)
        freeze_layer(net.bn)

    # def test method
    def test():  #evaluation

        test_acc  = utils.AverageMeter()
        test_loss = utils.AverageMeter()

        net.eval()
        print('Test started.')

        for i, data in enumerate(val_loader):

            states_test = net.init_hidden(is_train=False)  #안쓰는거 같음

            #if args.arch == 'Video' or args.arch == 'Audio':  # single modality
            if args.arch == 'Video' or args.arch == 'Video_fc'or args.arch == 'Audio':  # VIDEO & AUDIO MUL JSYDSHS  

                #input, target = data  # input is of shape torch.Size([batch, channels, frames, width, height])
                (input, input_audio, target) = data # VIDEO & AUDIO MUL JSYDSHS
                target_var = Variable(target.squeeze()).cuda()
                output = net(input.cuda(), states_test, input_audio.cuda())

            loss = criterion(output.squeeze(), target_var)

            # measure accuracy and record loss
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target.squeeze().cuda()).sum().type(torch.FloatTensor)
            accuracy.mul_((100.0 / args.test_batch_size))
            test_loss.update(loss.item(), args.test_batch_size)
            test_acc.update(accuracy.item(), args.test_batch_size)

            if i>0 and i % args.print_freq == 0:
                print('Test: [{0}][{1}/{2}] - loss = {3} , acc = {4}'.format(epoch, i, len(val_loader), test_loss.avg, test_acc.avg))

        net.train()
        print('Test finished.')
        return test_acc.avg, test_loss.avg


    ### main training loop ###

    best_accuracy = 0
    best_epoch = 0
    step = 0


    for epoch in range(0,args.num_epochs):

        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()

        # learning rate decay
        #scheduler.step()

        end = time.time()
        # train for one epoch
        for i, (input, input_audio ,target) in enumerate(train_loader):
            check4=time.time()
            states = net.init_hidden(is_train=True)  #안쓰는거 같음

            if args.arch == 'Video' or args.arch == 'Audio' or args.arch == 'Video_fc': # single modality  #args.arch = 'Video'만 씀 다른거 x

                target_var = Variable(target.squeeze()).cuda()
                check5=time.time()

                # measure data loading time
                data_time.update(time.time() - end)
                #pdb.set_trace()
                output = net(input.cuda().requires_grad_(True), states, input_audio.cuda().requires_grad_(True))
                check6=time.time()

            loss = criterion(output.squeeze(), target_var)
            # error 발생
            # compute gradient and do SGD step
            #pdb.set_trace()
            net.zero_grad()
            loss.backward()
            torchutils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure accuracy and record loss
            _, predicted = torch.max(output.data, 1)




            accuracy = (predicted == target.squeeze().cuda()).sum().type(torch.FloatTensor)
            accuracy.mul_((100.0 / args.batch_size))
            train_loss.update(loss.item(), args.batch_size)
            train_acc.update(accuracy.item(), args.batch_size)

            check=(time.time()-end)
#            print("check: {}".format(check))
            end=time.time()


            # tensorboard logging
            logger.scalar_summary('train loss', loss.item(), step + 1)
            logger.scalar_summary('train accuracy', accuracy.item(), step + 1)
            step+=1

            if i > 0 and i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}] , \t'
                      'LR {3} , \t'
                      'Time {batch_time.avg:.3f} , \t'
                      'Data {data_time.avg:.3f} , \t'
                      'Loss {loss.avg:.4f} , \t'
                      'Acc {top1.avg:.3f}'.format(
                      epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time, data_time=data_time, loss=train_loss, top1=train_acc))
        
        
        # evaluate on validation set
        accuracy, loss = test()                
        scheduler.step(loss)
        # logger
        logger.scalar_summary('Test Accuracy', accuracy, epoch)
        logger.scalar_summary('Test Loss ', loss, epoch)
        logger.scalar_summary('LR ', optimizer.param_groups[0]['lr'], epoch)

        # remember best prec@1 and save checkpoint
        is_best = False
        if accuracy > best_accuracy:
            is_best = True
            best_epoch = epoch

        best_accuracy = max(accuracy, best_accuracy)

        print('Average accuracy on validation set is: {}%'.format(accuracy))
        print('Best accuracy so far is: {}% , at epoch #{}'.format(best_accuracy,best_epoch))

        if epoch % args.save_freq == 0:
            checkpoint_name = "%s/acc_%.3f_epoch_%03d_arch_%s.pkl" % (save_dir, accuracy, epoch, args.arch)
            utils.save_checkpoint(state={
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best,best_accuracy=best_accuracy,filename=checkpoint_name)
            model_name = "%s/acc_%.3f_epoch_%03d_arch_%s_model.pkl" % (save_dir, accuracy, epoch, args.arch)

            torch.save(net, model_name)

            
        print("train finished")

