"""Video clip order prediction."""
import os
import math
import itertools
import argparse
import time
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
#from datasets.ucf101 import UCF101VCOPDataset
#from datasets.UCFjigsawFullPlusCLS import UCFJigsawveriffull,UCFJigsawverif8, PhotoJigsawverifall,PhotoJigsawverif8, UCFJigsaw,PhotoJigsawverifallflip,PhotoJigsawver8flip
from datasets.UCFjigsawFullPlusCLS15 import UCFJigsaw4all,PhotoJigsaw4all
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
#from models.opn import OPN
from models.modeljigsawfull import VCOPN
#from models.modeljigsawfull import VCOPN
#from models.vcopn import VCOPN
from models.alexnet import AlexNet
from PIL import Image


from shutil import copyfile
 
 
def load_pretrained_weights(ckpt_path):

    adjusted_weights = {};
    pretrained_weights = torch.load(ckpt_path,map_location='cpu');
    for name ,params in pretrained_weights.items():
        print(name)
        if "module.base_network" in name:
            name = name[name.find('.')+14:]
            adjusted_weights[name]=params;
    return adjusted_weights;

 
 
 
def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        #targets = [order_class_index(order,clslist) for order in tuple_orders]
        targets = tuple_orders.to(device)
        #targets = torch.tensor(targets).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        #print('a112')
        # compute loss and acc
        running_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        #print('a113')
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            #print('a115')
            avg_acc = correct / (args.pf * args.bs)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch-1)*len(train_dataloader) + i
            writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0
            #print('a3')
    # summary params and grads per eopch
    for name, param in model.named_parameters():
        writer.add_histogram('params/{}'.format(name), param, epoch)
        writer.add_histogram('grads/{}'.format(name), param.grad, epoch)

 
def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        #targets = [order_class_index(order,clslist) for order in tuple_orders]
        #targets = torch.tensor(targets).to(device)
        targets = tuple_orders.to(device)
        # forward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss,avg_acc


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        pts=[]
        targets=[]
        outputs=[]

        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        #targets = [order_class_index(order) for order in tuple_orders]
        #targets = torch.tensor(targets).to(device)
        targets = tuple_orders.to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        if(i%100==0):
          #torch.set_printoptions(profile="full")
          print('targ is',correct,i)
          #print('pts  is',pts)
          #print('out  is',outputs)
          #torch.set_printoptions(profile="default") 
           
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='test', help='train/test')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--it', type=int, default=4, help='interval')
    parser.add_argument('--tl', type=int, default=4, help='tuple length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str,default='home/o2/jigsaw/VCOP/jigsaw4/', help='log directory')
    parser.add_argument('--ckpt', type=str ,default='/content/drive/Shareddrives/ICLR/Jigsaw4orderpredict/best_model_428_0.017__0.985.pt',help='checkpoint path')#default= 
    #parser.add_argument('--ckpt', type=str,default='/content/drive/MyDrive/opnmodel/jigsaw4fullorder/jigsaw4fullordermodlall/best_model_138_0.165__0.932.pt',help='checkpoint path')#default= 
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=800, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=94, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=50, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=20, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print("device",device)
    print(torch.cuda.get_device_properties(device))
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)
    print('a0')
    ########### model ##############
    if args.model == 'c3d':
        base = C3D(with_classifier=False)
    elif args.model == 'r3d':
        base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    elif args.model == 'AlexNet':
        base = AlexNet(with_classifier=False, return_conv=False)
    elif args.model == 'r21d':   
        base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    #opn = OPN(base_network=base, feature_size=256, tuple_len=args.tl).to(device)
    opn = VCOPN(base_network=base, feature_size=256, tuple_len=args.tl).to(device)   
    multi_gpu=1
    if multi_gpu ==1:
        opn = nn.DataParallel(opn)
   
    if args.mode == 'train':  ########### Train #############
        if args.ckpt:  # resume training
            if args.start_epoch == 0     :
                pretrain_weight = load_pretrained_weights(args.ckpt)
                print(pretrain_weight.keys())
                opn.load_state_dict(pretrain_weight,strict=False)
                log_dir = os.path.dirname(args.ckpt)
            else:
                #print('happent')
                opn.load_state_dict(torch.load(args.ckpt))
                log_dir = os.path.dirname(args.ckpt)
            #print('1')
        else:
            if args.desp:
                exp_name = '{}_cl{}_it{}_tl{}_{}_{}'.format(args.model, args.cl, args.it, args.tl, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_cl{}_it{}_tl{}_{}'.format(args.model, args.cl, args.it, args.tl, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)
            #print('exp',exp_name)
        writer = SummaryWriter(log_dir)
        if args.tl==4 :
            train_transforms = transforms.Compose([
                transforms.Resize((57, 57)),
                transforms.CenterCrop(55),
                #transforms.RandomCrop(20,30),
                transforms.ToTensor()])            
        else:
                train_transforms = transforms.Compose([
                transforms.Resize((38, 38)),
                transforms.CenterCrop(36),
                #transforms.RandomCrop(20,30),
                transforms.ToTensor()])
            
 
        clslist = np.load('sol2.npy')
        #train_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, True, train_transforms)
        print('a')
        # verif
        tl=4
        # train_dataset = UCFJigsawveriffull('data/ucf101', args.cl, args.it, '012',  tl,  True,train_transforms  )
        # val_dataset = UCFJigsawveriffull('data/ucf101', args.cl, args.it, '011',  tl,  True,train_transforms  )
        # # full4
        train_dataset = UCFJigsaw4all('data/ucf101', args.cl, args.it, '012',  tl,  True,train_transforms  )
        val_dataset = UCFJigsaw4all('data/ucf101', args.cl, args.it, '011',  tl,  True,train_transforms  )
    
        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)
        #print('train_dataloader',train_dataloader)


        #################
        
        
        
        
        
        
        if args.ckpt:
            pass
        else:
            # save graph and clips_order samples
            for data in train_dataloader:
                #print('data1 ')
                tuple_frame, tuple_orders = data

                # for i in range(args.tl):
                #         writer.add_images('train/tuple_frame', tuple_frame[:, i, :, :, :], i)
                #         writer.add_text('train/tuple_orders', str(tuple_orders[i].tolist()), i)
                #tuple_clips = tuple_clips.to(device)
                tuple_frame = tuple_frame.to(device)
                print('tps',tuple_frame.size())
                #writer.add_graph(opn, tuple_frame,verbose=False)
                #writer.add_graph(opn, tuple_frame)
                #writer.flush()
                #writer.close()
                break
            # save init params at step 0
            for name, param in opn.named_parameters():
                writer.add_histogram('params/{}'.format(name), param, 0)

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(opn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

        prev_best_val_loss = float('inf')
        prev_best_val_acc=0
        val_loss = float('inf')
        val_acc=0
        prev_best_model_path = None
        model_path_prev=None
        print('here i am')
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            if args.start_epoch==epoch:
              time_start = time.time()
            train(args, opn, criterion, optimizer, device, train_dataloader, writer, epoch)
            if args.start_epoch==epoch:
              print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            # scheduler.step(val_loss)         
            val_loss,val_acc = validate(args, opn, criterion, device, val_dataloader, writer, epoch)
            # scheduler.step(val_loss)         
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)            # save model every 1 epoches
            if epoch % 1 == 0:
              torch.save(opn.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
              if  model_path_prev:
                  open(model_path_prev, 'w').close() #overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                  os.remove(model_path_prev) #delete the blank file from google drive will move the file to bin instead
              model_path_prev = os.path.join(log_dir, 'model_{}.pt'.format(epoch))
               

            # save model for the best val
            if val_loss < prev_best_val_loss:
                model_path = os.path.join(log_dir, 'best_model_{}_{:.3f}__{:.3f}.pt'.format(epoch,val_loss,val_acc))
                torch.save(opn.state_dict(), model_path)
                prev_best_val_loss = val_loss
                #if prev_best_model_path:
                    #os.remove(prev_best_model_path)
                prev_best_model_path = model_path
                if(val_acc > prev_best_val_acc):
                  prev_best_val_acc = val_acc
            elif(val_acc > prev_best_val_acc):
                model_path = os.path.join(log_dir, 'best_model_{}_{:.3f}_{:.3f}.pt'.format(epoch,val_loss,val_acc))
                torch.save(opn.state_dict(), model_path)
                prev_best_val_acc = val_acc
                #if prev_best_model_path:
                    #os.remove(prev_best_model_path)
                prev_best_model_path = model_path

    elif args.mode == 'test':  ########### Test #############
        #model1 = torch.load('model_1.pt')
        #model.load_state_dict(model1)
        opn.load_state_dict(torch.load(args.ckpt))
        #opn.load_state_dict(model1)
        test_transforms = transforms.Compose([
                transforms.Resize((57, 57)),
                transforms.CenterCrop(55),
                #transforms.RandomCrop(20,30),
                transforms.ToTensor()])   
       
        #train_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, True, train_transforms)
        print('a')
        tl=9
        #test_dataset = UCF101VCOPDataset('data/ucf101', args.cl, args.it, args.tl, False, test_transforms)
        #test_dataset = UCF101FOPDataset('data/ucf101',  args.it, args.tl, False, test_transforms)
        #test_dataset = UCFJigsaw('data/ucf101', args.cl, args.it, '013',  tl,  True,test_transforms  )
        #test_dataset = UCF101FOPDatasetphoto('data/photos',  args.it, args.tl, False, test_transforms)
        #test_dataset = PhotoJigsawverifallflip('data/images/tennis', args.cl, args.it, 'tennis',  tl,  False,test_transforms  )
        #test_dataset = PhotoJigsawverifall('data/images/retme', args.cl, args.it, 'retme',  tl,  False,test_transforms  )
        #test_dataset = PhotoJigsawver8flip('data/images/retme', args.cl, args.it, 'retme',  tl,  False,test_transforms  )
        #test_dataset = PhotoJigsaw4all('data/images/retme', args.cl, args.it, 'retme',  tl,  False,test_transforms  )
        #test_dataset = UCFJigsawverif8('data/images/retme', args.cl, args.it, 'retme',  tl,  False,test_transforms  )
        #test_dataset = UCFJigsaw4all('data/ucf101', args.cl, args.it, '013',  tl,  True,test_transforms  )
        #test_dataset = PhotoJigsaw4all('/content/drive/Shareddrives/ICLR/Dataset/test2014COCO', args.cl, args.it, '/content/drive/Shareddrives/ICLR/Dataset/COCO2',  tl,  False,test_transforms  )
        test_dataset = PhotoJigsaw4all('/content/drive/Shareddrives/ICLR/Dataset', args.cl, args.it, '/content/drive/Shareddrives/ICLR/Dataset/retargetmefull',  tl,  False,test_transforms  )

        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, opn, criterion, device, test_dataloader)