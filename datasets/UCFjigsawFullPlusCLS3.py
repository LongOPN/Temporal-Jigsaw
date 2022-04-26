"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile
import itertools
import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
#from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
import skimage.io
from skimage import io
from random import choice




class photoJigsaw9allrotation(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(9))))
      
        #self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]



    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        
        filename = os.path.join(self.root_dir,  videoname)
        #self.set = pd.read_csv(filename, header=None)[0]
 
        img = img=io.imread(filename,0)
                             
        img=      cv2.flip(img,-1)

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((128, 171)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        verif = random.randint(0, 1)
  
        if verif==1:
            img=      cv2.flip(img,-1)


        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
        clip_and_order = list(zip(tiles, tuple_order))
 
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []
        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
 
        cls = self.classes.index(tuple(tuple_order))
      
        return torch.stack(tuple_frame), torch.tensor(int(verif))  
 



class UCFJigsaw9allrotation(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(9))))
      
        #self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        
        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')
          

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((128, 171)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        verif = random.randint(0, 1)
  
        if verif==1:
            img=      cv2.flip(img,-1)


        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
        clip_and_order = list(zip(tiles, tuple_order))
 
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []
        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
 
        cls = self.classes.index(tuple(tuple_order))
      
        return torch.stack(tuple_frame), torch.tensor(int(verif))  
 





class UCF101Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, clip_len, split='1', train=True, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
       

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        # random select a clip for train
        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        # sample several clips for test
        else:
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                clip_start = int(i - self.clip_len/2)
                clip = videodata[clip_start: clip_start + self.clip_len]
                if self.transforms_:
                    trans_clip = []
                    # fix seed, apply the sample `random transformation` for all frames in the clip 
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame) # PIL image
                        frame = self.transforms_(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))

            return torch.stack(all_clips), torch.tensor(int(class_idx))



class PhotoJigsawverifallflip(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
 
        clsp=[]
        ac=0
        bc=12996
        cc=85949
        dc=105306
        ec=257573 
        fc= 276930
        gc=349883
        hc=362879
        clsp.append(ac)
        clsp.append(bc)
        clsp.append(cc)
        clsp.append(dc)
        clsp.append(ec)
        clsp.append(fc)
        clsp.append(gc)
        clsp.append(hc)
        self.clsp=clsp
        
        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
          

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir,  videoname)
        #self.set = pd.read_csv(filename, header=None)[0]
 
        img = img=io.imread(filename,0)
                             
        img=      cv2.flip(img,-1)
           
            
        



        tuple_clip = []
 
            
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)
  
        #s=38
        s = float(img.size[0]) // 3  
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9): 
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
       
               
      
         
        classes = list(itertools.permutations(list(range(9))))

        per=[]
        seed = random.random()
        verif = random.randint(0, 1)
  
        if verif==0:
            tp = random.randint(0, 7)    
            #print(tp)    

            p= self.clsp[0]
            tuple_order= classes[p]
            #tuple_order= list(per)
             
        else:     
            tf=choice([i for i in range(0,362879) if i not in [self.clsp]])           
            tuple_order= classes[tf]        
            #tuple_order= list(per)
      
        tuple_clip=[]
        for i in tuple_order:   
           tuple_clip.append(tiles[i])

     

            
       # tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []

        # # random shuffle for train, the same shuffle for test
        # if self.train:
        #     random.shuffle(clip_and_order)
        # else:
        #     random.seed(idx)
        #     random.shuffle(clip_and_order)

        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
        # else:
        #     tuple_frame = [torch.tensor(frame) for frame in tuple_frame]

        return torch.stack(tuple_frame),  torch.tensor(int(verif))




class PhotoJigsawver8flip(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
 
        clsp=[]
        ac=0
        bc=12996
        cc=85949
        dc=105306
        ec=257573 
        fc= 276930
        gc=349883
        hc=362879
        clsp.append(ac)
        clsp.append(bc)
        clsp.append(cc)
        clsp.append(dc)
        clsp.append(ec)
        clsp.append(fc)
        clsp.append(gc)
        clsp.append(hc)
        self.clsp=clsp
        
        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
          

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir,  videoname)
        #self.set = pd.read_csv(filename, header=None)[0]
 
        img = img=io.imread(filename,0)
                             
        img=      cv2.flip(img,-1)
           
            
        



        tuple_clip = []
 
            
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)
  
        #s=38
        s = float(img.size[0]) // 3  
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9): 
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tiles.append(tile)
       
                    
         
        classes = list(itertools.permutations(list(range(9))))

        per=[]
        seed = random.random()
        verif = random.randint(0, 1)
  
        if verif==0:
            tp = random.randint(0, 7)    
            #print(tp)    

            p= self.clsp[tp]
            tuple_order= classes[p]
            #tuple_order= list(per)
             
        else:     
            tf=choice([i for i in range(0,362879) if i not in [self.clsp]])           
            tuple_order= classes[tf]        
            #tuple_order= list(per)
      
        tuple_clip=[]
        for i in tuple_order:   
           tuple_clip.append(tiles[i])

            
       # tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []

        # # random shuffle for train, the same shuffle for test
        # if self.train:
        #     random.shuffle(clip_and_order)
        # else:
        #     random.seed(idx)
        #     random.shuffle(clip_and_order)

        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
        # else:
        #     tuple_frame = [torch.tensor(frame) for frame in tuple_frame]

        return torch.stack(tuple_frame),  torch.tensor(int(verif))




class PhotoJigsawverifall(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
 
        clsp=[]
        ac=0
        bc=12996
        cc=85949
        dc=105306
        ec=257573 
        fc= 276930
        gc=349883
        hc=362879
        clsp.append(ac)
        clsp.append(bc)
        clsp.append(cc)
        clsp.append(dc)
        clsp.append(ec)
        clsp.append(fc)
        clsp.append(gc)
        clsp.append(hc)
        self.clsp=clsp
        
        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
          

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir,  videoname)
        #self.set = pd.read_csv(filename, header=None)[0]
 
        img = img=io.imread(filename,0)
        

        tuple_clip = []
 
            
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)
  
        #s=38
        s = float(img.size[0]) // 3  
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9): 
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
          
         
        classes = list(itertools.permutations(list(range(9))))

        per=[]
        seed = random.random()
        verif = random.randint(0, 1)
  
        if verif==0:
            tp = random.randint(0, 7)    
            #print(tp)    

            p= self.clsp[0]
            tuple_order= classes[p]
            #tuple_order= list(per)
             
        else:     
            tf=choice([i for i in range(0,362879) if i not in [self.clsp]])           
            tuple_order= classes[tf]        
            #tuple_order= list(per)
            
        tuple_clip=[]
        for i in tuple_order:   
          tuple_clip.append(tiles[i])
              
       # tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []

        # # random shuffle for train, the same shuffle for test
        # if self.train:
        #     random.shuffle(clip_and_order)
        # else:
        #     random.seed(idx)
        #     random.shuffle(clip_and_order)

        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
        # else:
        #     tuple_frame = [torch.tensor(frame) for frame in tuple_frame]

        return torch.stack(tuple_frame),  torch.tensor(int(verif))





class PhotoJigsawverif8(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
 
        clsp=[]
        ac=0
        hc=362879
        bc=12996
        cc=85949
        dc=105306
        ec=257573 
        fc= 276930
        gc=349883
        
        clsp.append(ac)
        clsp.append(hc)
        clsp.append(bc)
        clsp.append(cc)
        clsp.append(dc)
        clsp.append(ec)
        clsp.append(fc)
        clsp.append(gc)
       
        self.clsp=clsp
        
        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        #while (True)  :
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        

        filename = os.path.join(self.root_dir, videoname)
        #self.set = pd.read_csv(filename, header=None)[0]

        img = img=io.imread(filename,0)
        #skimage.io.imread(img,0)

        # if len(img.shape)==3: 
        #     break;
        # else:
        #     idx=idx+1
        #     print(filename)
                        

    
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

        tuple_clip = []
 
        
  
        #s=38
        s = float(img.size[0]) // 3  
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9): 
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            

            tiles.append(tile)
        seed = random.random()
        verif = random.randint(0, 1)            
        #clip_and_order = list(zip(tiles, tuple_order))
         
        classes = list(itertools.permutations(list(range(9))))

        per=[]
      
      
  
        if verif==0:
            tp = random.randint(0, 1)    
            #print(tp)    
            p= self.clsp[tp]
            tuple_order= classes[p]
            #tuple_order= list(per)
             
        else:     
            tf=choice([i for i in range(0,362879) if i not in [self.clsp]])           
            tuple_order= classes[tf]        
            #tuple_order= list(per)
            
        tuple_clip=[]
        for i in tuple_order:   
          tuple_clip.append(tiles[i])
              
       # tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []

        # # random shuffle for train, the same shuffle for test
        # if self.train:
        #     random.shuffle(clip_and_order)
        # else:
        #     random.seed(idx)
        #     random.shuffle(clip_and_order)

        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
        # else:
        #     tuple_frame = [torch.tensor(frame) for frame in tuple_frame]

        return torch.stack(tuple_frame),  torch.tensor(int(verif))



class Jigsawphoto(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(9))))
        self.clslist = np.load('sol2.npy')
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]        
        filename = os.path.join(self.root_dir,  videoname)
        #self.set = pd.read_csv(filename, header=None)[0]
 
        img = img=io.imread(filename)
        

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
    
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
        clip_and_order = list(zip(tiles, tuple_order))
 
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []
        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
        # else:
        #     tuple_frame = [torch.tensor(frame) for frame in tuple_frame]
        

        a= tuple_order
        classes=self.classes
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        a5=[]
        a6=[]
        a7=[]
      
        for i in range(3):
          #print(a[i])
          j=i%3 
          for k1 in [8,5,2]:
            a1.append(a[k1-j])
          for k2 in [6,3,0]:
            a2.append(a[k2+j]) 
          for m5 in [0,3,6]  :
              a6.append(a[m5+j])  
          for m6 in [2,5,8]  :
              a7.append(a[m6-j])  

        for k3 in [8,5,2]  :
          for k4 in range(3):
              a3.append(a[k3-k4])
        for k5 in [6,3,0]  :
          for k6 in range(3):
              a4.append(a[k5+k6])  
                  
        for m3 in [2,5,8]  :
          for m4 in range(3):
              a5.append(a[m3-m4])

        #print( '\n',a1,'\n',a2,'\n',a3,'\n',a4,'\n',a5,'\n',a6,'\n',a7)
        # print(a, classes.index(tuple(a)))
        # print(a1, classes.index(tuple(a1)))
        # print(a2, classes.index(tuple(a2)))
        # print(a3, classes.index(tuple(a3)))
        # print(a4, classes.index(tuple(a4)))
        # print(a5, classes.index(tuple(a5)))
        # print(a6, classes.index(tuple(a6)))
        # print(a7, classes.index(tuple(a7)))
        minc=100000000000
        if (minc>= classes.index(tuple(a))):
          minc= classes.index(tuple(a))
        if (minc>= classes.index(tuple(a1))):
          minc= classes.index(tuple(a1))
        if (minc>= classes.index(tuple(a2))):
          minc= classes.index(tuple(a2))
        if (minc>= classes.index(tuple(a3))):
          minc= classes.index(tuple(a3))
        if (minc>= classes.index(tuple(a4))):
          minc= classes.index(tuple(a4))
        if (minc>= classes.index(tuple(a5))):
          minc= classes.index(tuple(a5))
        if (minc>= classes.index(tuple(a6))):
          minc= classes.index(tuple(a6))
        if (minc>= classes.index(tuple(a7))):
          minc= classes.index(tuple(a7))

        temp=self.temp
        cls = temp.tolist().index(minc)

        return torch.stack(tuple_frame), torch.tensor(int(cls))  




class UCFJigsawverif8(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(9))))
        self.clslist = np.load('sol2.npy')
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        clsp=[]
        ac=0
        hc=362879
        bc=12996
        cc=85949
        dc=105306
        ec=257573 
        fc= 276930
        gc=349883
        
        clsp.append(ac)
        clsp.append(hc)
        clsp.append(bc)
        clsp.append(cc)
        clsp.append(dc)
        clsp.append(ec)
        clsp.append(fc)
        clsp.append(gc)
       
        self.clsp=clsp


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
      
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)           
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)

        seed = random.random()
        verif = random.randint(0, 1)            
        #clip_and_order = list(zip(tiles, tuple_order))
         
        classes = list(itertools.permutations(list(range(9))))

        per=[]
      
      
  
        if verif==0:
            tp = random.randint(0, 1)    
            #print(tp)    
            p= self.clsp[tp]
            tuple_order= classes[p]
            #tuple_order= list(per)
             
        else:     
            tf=choice([i for i in range(0,362879) if i not in [self.clsp]])           
            tuple_order= classes[tf]        
            #tuple_order= list(per)
            
        tuple_clip=[]
        for i in tuple_order:   
          tuple_clip.append(tiles[i])
              
       # tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []


        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
      
 

        return torch.stack(tuple_frame), torch.tensor(int(verif))  


class UCFJigsawverif8pthrot(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(9))))
        self.clslist = np.load('sol2.npy')
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        clsp=[]
        ac=0
        hc=362879
        bc=12996
        cc=85949
        dc=105306
        ec=257573 
        fc= 276930
        gc=349883
        
        clsp.append(ac)
        clsp.append(hc)
        clsp.append(bc)
        clsp.append(cc)
        clsp.append(dc)
        clsp.append(ec)
        clsp.append(fc)
        clsp.append(gc)
       
        self.clsp=clsp


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
      
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)           
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile=      cv2.flip(tile,-1)
            tiles.append(tile)

        seed = random.random()
        verif = random.randint(0, 1)            
        #clip_and_order = list(zip(tiles, tuple_order))
         
        classes = list(itertools.permutations(list(range(9))))

        per=[]
      
      
  
        if verif==0:
            tp = random.randint(0, 1)    
            #print(tp)    
            p= self.clsp[tp]
            tuple_order= classes[p]
            #tuple_order= list(per)
             
        else:     
            tf=choice([i for i in range(0,362879) if i not in [self.clsp]])           
            tuple_order= classes[tf]        
            #tuple_order= list(per)
            
        tuple_clip=[]
        for i in tuple_order:   
          tuple_clip.append(tiles[i])
              
       # tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []


        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
      
 

        return torch.stack(tuple_frame), torch.tensor(int(verif))  





class UCFJigsawveriffull(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(9))))
        self.clslist = np.load('sol2.npy')
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        clsp=[]
        ac=0
        hc=362879
        bc=12996
        cc=85949
        dc=105306
        ec=257573 
        fc= 276930
        gc=349883
        
        clsp.append(ac)
        clsp.append(hc)
        clsp.append(bc)
        clsp.append(cc)
        clsp.append(dc)
        clsp.append(ec)
        clsp.append(fc)
        clsp.append(gc)
       
        self.clsp=clsp


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)           
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)

        seed = random.random()
        verif = random.randint(0, 1)            
        #clip_and_order = list(zip(tiles, tuple_order))
         
        classes = list(itertools.permutations(list(range(9))))

        per=[]
      
      
  
        if verif==0:
            tp = random.randint(0, 1)    
            #print(tp)    
            p= self.clsp[0]
            tuple_order= classes[p]
            #tuple_order= list(per)
             
        else:     
            tf=choice([i for i in range(0,362879) if i not in [self.clsp]])           
            tuple_order= classes[tf]        
            #tuple_order= list(per)
            
        tuple_clip=[]
        for i in tuple_order:   
          tuple_clip.append(tiles[i])
              
       # tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []


        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
      
 

        return torch.stack(tuple_frame), torch.tensor(int(verif))  




class UCFJigsaw(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(9))))
        self.clslist = np.load('sol2.npy')
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((148, 191)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
        clip_and_order = list(zip(tiles, tuple_order))
 
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []
        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
        # else:
        #     tuple_frame = [torch.tensor(frame) for frame in tuple_frame]
        

        a= tuple_order
        classes=self.classes
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        a5=[]
        a6=[]
        a7=[]
      
        for i in range(3):
          #print(a[i])
          j=i%3 
          for k1 in [8,5,2]:
            a1.append(a[k1-j])
          for k2 in [6,3,0]:
            a2.append(a[k2+j]) 
          for m5 in [0,3,6]  :
              a6.append(a[m5+j])  
          for m6 in [2,5,8]  :
              a7.append(a[m6-j])  

        for k3 in [8,5,2]  :
          for k4 in range(3):
              a3.append(a[k3-k4])
        for k5 in [6,3,0]  :
          for k6 in range(3):
              a4.append(a[k5+k6])  
                  
        for m3 in [2,5,8]  :
          for m4 in range(3):
              a5.append(a[m3-m4])

        #print( '\n',a1,'\n',a2,'\n',a3,'\n',a4,'\n',a5,'\n',a6,'\n',a7)
        # print(a, classes.index(tuple(a)))
        # print(a1, classes.index(tuple(a1)))
        # print(a2, classes.index(tuple(a2)))
        # print(a3, classes.index(tuple(a3)))
        # print(a4, classes.index(tuple(a4)))
        # print(a5, classes.index(tuple(a5)))
        # print(a6, classes.index(tuple(a6)))
        # print(a7, classes.index(tuple(a7)))
        minc=100000000000
        if (minc>= classes.index(tuple(a))):
          minc= classes.index(tuple(a))
        if (minc>= classes.index(tuple(a1))):
          minc= classes.index(tuple(a1))
        if (minc>= classes.index(tuple(a2))):
          minc= classes.index(tuple(a2))
        if (minc>= classes.index(tuple(a3))):
          minc= classes.index(tuple(a3))
        if (minc>= classes.index(tuple(a4))):
          minc= classes.index(tuple(a4))
        if (minc>= classes.index(tuple(a5))):
          minc= classes.index(tuple(a5))
        if (minc>= classes.index(tuple(a6))):
          minc= classes.index(tuple(a6))
        if (minc>= classes.index(tuple(a7))):
          minc= classes.index(tuple(a7))
    



        temp=self.temp
        cls = temp.tolist().index(minc)

        return torch.stack(tuple_frame), torch.tensor(int(cls))  


class UCF101Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, clip_len, split='1', train=True, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
       

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        # random select a clip for train
        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        # sample several clips for test
        else:
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                clip_start = int(i - self.clip_len/2)
                clip = videodata[clip_start: clip_start + self.clip_len]
                if self.transforms_:
                    trans_clip = []
                    # fix seed, apply the sample `random transformation` for all frames in the clip 
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame) # PIL image
                        frame = self.transforms_(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))

            return torch.stack(all_clips), torch.tensor(int(class_idx))


 


class UCFJigsaw4(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=4, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(4))))
        self.clslist = np.load('soljig4.npy')
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((128, 171)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 2
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(4):
            i = n // 2
            j = n % 2
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
        clip_and_order = list(zip(tiles, tuple_order))
 
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []
        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
        # else:
        #     tuple_frame = [torch.tensor(frame) for frame in tuple_frame]
        

        a= tuple_order
        classes=self.classes
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        a5=[]
        a6=[]
        a7=[]
              
        for i in range(2):
          #print(a[i])
          j=i%2 
          for k1 in [3,1]:
            a1.append(a[k1-j])
          for k2 in [2,0]:
            a2.append(a[k2+j]) 
          for m5 in [0,2]  :
              a6.append(a[m5+j])  
          for m6 in [1,3]  :
              a7.append(a[m6-j])  

        for k3 in [3,1]  :
          for k4 in range(2):
              a3.append(a[k3-k4])
        for k5 in [2,0]  :
          for k6 in range(2):
              a4.append(a[k5+k6])  
                  
        for m3 in [1,3]  :
          for m4 in range(2):
              a5.append(a[m3-m4])
          
        minc=100000000000
        if (minc>= classes.index(tuple(a))):
          minc= classes.index(tuple(a))
        if (minc>= classes.index(tuple(a1))):
          minc= classes.index(tuple(a1))
        if (minc>= classes.index(tuple(a2))):
          minc= classes.index(tuple(a2))
        if (minc>= classes.index(tuple(a3))):
          minc= classes.index(tuple(a3))
        if (minc>= classes.index(tuple(a4))):
          minc= classes.index(tuple(a4))
        if (minc>= classes.index(tuple(a5))):
          minc= classes.index(tuple(a5))
        if (minc>= classes.index(tuple(a6))):
          minc= classes.index(tuple(a6))
        if (minc>= classes.index(tuple(a7))):
          minc= classes.index(tuple(a7))
    


        
        temp=self.temp
        cls = temp.tolist().index(minc)
      
        return torch.stack(tuple_frame), torch.tensor(int(cls)) , torch.tensor(tuple_order)
 



class UCFJigsaw4all(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=4, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(4))))
      
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((128, 171)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 2
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(4):
            i = n // 2
            j = n % 2
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
        clip_and_order = list(zip(tiles, tuple_order))
 
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []
        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
 
        cls = classes.index(tuple(tuple_order))
      
        return torch.stack(tuple_frame), torch.tensor(int(cls))  
 



class UCFJigsaw9all(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=4, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(4))))
      
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((128, 171)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tiles.append(tile)
        clip_and_order = list(zip(tiles, tuple_order))
 
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []
        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
 
        cls = classes.index(tuple(tuple_order))
      
        return torch.stack(tuple_frame), torch.tensor(int(cls))  
 



class UCFJigsaw6(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval,split='1', tuple_len=9, train=True, transforms_=None ):
  
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.split= split
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.classes =    list(itertools.permutations(list(range(9))))
        self.clslist = np.load('sol2.npy')
        self.temp = np.array(self.clslist) # NumPy array
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)


        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split = os.path.join(root_dir, 'split', 'trainlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split, header=None, sep=' ')[0]
            test_split = os.path.join(root_dir, 'split', 'testlist01.txt')

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        self.tuple_len =6
        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(1, length-2)
        else:
            random.seed(idx)
            tuple_start = random.randint(1, length-2)
        image_transformer = transforms.Compose([
                            transforms.Resize((138, 191)),
                            transforms.CenterCrop(114),
                            ])
        img = videodata[tuple_start]
        img = self.toPIL(img) # PIL image
        img= image_transformer(img)

 
        s = float(img.size[0]) // 3
        a = s // 2
        #tiles = [None] * 9
        tiles=[]
        for n in range(9):
            i = n // 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            if n<=6:
                tiles.append(tile)
         
         
        clip_and_order = list(zip(tiles, tuple_order))
 
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)
        trans_tuple = []
        # if self.transforms_:
        trans_tuple = []
        for frame in tuple_clip:
            # frame = self.toPIL(frame) # PIL image
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_tuple.append(frame)
        tuple_frame = trans_tuple
        # else:
        #     tuple_frame = [torch.tensor(frame) for frame in tuple_frame]
        

        a= tuple_order
        classes=self.classes
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        a5=[]
        a6=[]
        a7=[]
      
        for i in range(3):
          #print(a[i])
          j=i%3 
          for k1 in [8,5,2]:
            a1.append(a[k1-j])
          for k2 in [6,3,0]:
            a2.append(a[k2+j]) 
          for m5 in [0,3,6]  :
              a6.append(a[m5+j])  
          for m6 in [2,5,8]  :
              a7.append(a[m6-j])  

        for k3 in [8,5,2]  :
          for k4 in range(3):
              a3.append(a[k3-k4])
        for k5 in [6,3,0]  :
          for k6 in range(3):
              a4.append(a[k5+k6])  
                  
        for m3 in [2,5,8]  :
          for m4 in range(3):
              a5.append(a[m3-m4])

        #print( '\n',a1,'\n',a2,'\n',a3,'\n',a4,'\n',a5,'\n',a6,'\n',a7)
        # print(a, classes.index(tuple(a)))
        # print(a1, classes.index(tuple(a1)))
        # print(a2, classes.index(tuple(a2)))
        # print(a3, classes.index(tuple(a3)))
        # print(a4, classes.index(tuple(a4)))
        # print(a5, classes.index(tuple(a5)))
        # print(a6, classes.index(tuple(a6)))
        # print(a7, classes.index(tuple(a7)))
        minc=100000000000
        if (minc>= classes.index(tuple(a))):
          minc= classes.index(tuple(a))
        if (minc>= classes.index(tuple(a1))):
          minc= classes.index(tuple(a1))
        if (minc>= classes.index(tuple(a2))):
          minc= classes.index(tuple(a2))
        if (minc>= classes.index(tuple(a3))):
          minc= classes.index(tuple(a3))
        if (minc>= classes.index(tuple(a4))):
          minc= classes.index(tuple(a4))
        if (minc>= classes.index(tuple(a5))):
          minc= classes.index(tuple(a5))
        if (minc>= classes.index(tuple(a6))):
          minc= classes.index(tuple(a6))
        if (minc>= classes.index(tuple(a7))):
          minc= classes.index(tuple(a7))
    
        newcls= classes.index(tuple(tuple_order))


        temp=self.temp
        cls = temp.tolist().index(minc)

        return torch.stack(tuple_frame), torch.tensor(int(newcls))  


 
class UCF101ClipRetrievalDataset(Dataset):
    """UCF101 dataset for Retrieval. Sample clips for each video. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, sample_num, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.sample_num = sample_num
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist01.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist01.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        all_clips = []
        all_idx = []
        for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.sample_num):
            clip_start = int(i - self.clip_len/2)
            clip = videodata[clip_start: clip_start + self.clip_len]
            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)
            all_clips.append(clip)
            all_idx.append(torch.tensor(int(class_idx)))

        return torch.stack(all_clips), torch.stack(all_idx)


class UCF101VCOPDataset(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        if self.train:
            vcop_train_split_name = 'vcop_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_train_split_path = os.path.join(root_dir, 'split', vcop_train_split_name)
            self.train_split = pd.read_csv(vcop_train_split_path, header=None)[0]
        else:
            vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_test_split_path = os.path.join(root_dir, 'split', vcop_test_split_name)
            self.test_split = pd.read_csv(vcop_test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip = videodata[clip_start: clip_start + self.clip_len]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip), torch.tensor(tuple_order)


class UCF101FrameRetrievalDataset(Dataset):
    """UCF101 dataset for Retrieval. Sample frames for each video. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, sample_num, train=True, transforms_=None):
        self.root_dir = root_dir
        self.sample_num = sample_num
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            vcop_train_split_name = 'trainlist0{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_train_split_path = os.path.join(root_dir, 'split', vcop_train_split_name)

            train_split_path = os.path.join(root_dir, 'split', 'trainlist01.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist01.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        
        all_frames = []
        all_idx = []
        for i in np.linspace(0, length-1, self.sample_num):
            frame = videodata[int(i)]
            if self.transforms_:
                frame = self.toPIL(frame) # PIL image
                frame = self.transforms_(frame) # tensor [C x H x W]
            else:
                frame = torch.tensor(frame) 
            all_frames.append(frame)
            all_idx.append(torch.tensor(int(class_idx)))

        return torch.stack(all_frames), torch.stack(all_idx)







  
 
class UCF101FOPDataset(Dataset):
    """UCF101 dataset for frame order prediction. Generate frames and permutes them on-the-fly. 
    May corrupt if there exists video which is not long enough.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        interval (int): number of frames between selected frames.
        tuple_len (int): number of selected frames in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, interval, tuple_len, train=True, transforms_=None):
        self.root_dir = root_dir
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = tuple_len + interval * (tuple_len - 1)

        if self.train:
            train_split_path = os.path.join(root_dir, 'split', 'trainlist01.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'split', 'testlist01.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_frame (tensor): [tuple_len x channel x height x width]
            tuple_order (tensor): [tuple_len]
        """

       
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]        
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape

        tuple_frame = []
        tuple_order = list(range(0, self.tuple_len))
        
        # random select frame for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)

        frame_idx = tuple_start
        for _ in range(self.tuple_len):
            tuple_frame.append(videodata[frame_idx])
            frame_idx = frame_idx + self.interval

        frame_and_order = list(zip(tuple_frame, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(frame_and_order)
        else:
            random.seed(idx)
            random.shuffle(frame_and_order)
        tuple_frame, tuple_order = zip(*frame_and_order)

        if self.transforms_:
            trans_tuple = []
            for frame in tuple_frame:
                frame = self.toPIL(frame) # PIL image
                frame = self.transforms_(frame) # tensor [C x H x W]
                trans_tuple.append(frame)
            tuple_frame = trans_tuple
        else:
            tuple_frame = [torch.tensor(frame) for frame in tuple_frame]

        return torch.stack(tuple_frame), torch.tensor(tuple_order)


def export_tuple(tuple_clip, tuple_order, dir):
    """export tuple_clip and set its name with correct order.
    
    Args:
        tuple_clip (tensor): [tuple_len x channel x time x height x width]
        tuple_order (tensor): [tuple_len]
    """
    tuple_len, channel, time, height, width = tuple_clip.shape
    for i in range(tuple_len):
        filename = os.path.join(dir, 'c{}.mp4'.format(tuple_order[i]))
        skvideo.io.vwrite(filename, tuple_clip[i])


def gen_ucf101_vcop_splits(root_dir, clip_len, interval, tuple_len):
    """Generate split files for different configs."""
    vcop_train_split_name = 'vcop_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    vcop_train_split_path = os.path.join(root_dir, 'split', vcop_train_split_name)
    vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
    vcop_test_split_path = os.path.join(root_dir, 'split', vcop_test_split_name)
    # minimum length of video to extract one tuple
    min_video_len = clip_len * tuple_len + interval * (tuple_len - 1)

    def _video_longer_enough(filename):
        """Return true if video `filename` is longer than `min_video_len`"""
        path = os.path.join(root_dir, 'video', filename)
        metadata = ffprobe(path)['video']
        return eval(metadata['@nb_frames']) >= min_video_len

    train_split = pd.read_csv(os.path.join(root_dir, 'split', 'trainlist01.txt'), header=None, sep=' ')[0]
    train_split = train_split[train_split.apply(_video_longer_enough)]
    train_split.to_csv(vcop_train_split_path, index=None)

    test_split = pd.read_csv(os.path.join(root_dir, 'split', 'testlist01.txt'), header=None, sep=' ')[0]
    test_split = test_split[test_split.apply(_video_longer_enough)]
    test_split.to_csv(vcop_test_split_path, index=None)


def ucf101_stats():
    """UCF101 statistics"""
    collects = {'nb_frames': [], 'heights': [], 'widths': [], 
                'aspect_ratios': [], 'frame_rates': []}

    for filename in glob('../data/ucf101/video/*/*.avi'):
        metadata = ffprobe(filename)['video']
        collects['nb_frames'].append(eval(metadata['@nb_frames']))
        collects['heights'].append(eval(metadata['@height']))
        collects['widths'].append(eval(metadata['@width']))
        collects['aspect_ratios'].append(metadata['@display_aspect_ratio'])
        collects['frame_rates'].append(eval(metadata['@avg_frame_rate']))

    stats = {key: sorted(list(set(collects[key]))) for key in collects.keys()}
    stats['nb_frames'] = [stats['nb_frames'][0], stats['nb_frames'][-1]]

    #pprint(stats)


if __name__ == '__main__':
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ucf101_stats()
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 16, 2)
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 32, 3)
    gen_ucf101_vcop_splits('../data/ucf101', 16, 8, 3)

    # train_transforms = transforms.Compose([
    #     transforms.Resize((128, 171)),
    #     transforms.RandomCrop(112),
    #     transforms.ToTensor()])
    # # train_dataset = UCF101FOPDataset('../data/ucf101', 8, 3, True, train_transforms)
    # # train_dataset = UCF101VCOPDataset('../data/ucf101', 16, 8, 3, True, train_transforms)
    # train_dataset = UCF101Dataset('../data/ucf101', 16, False, train_transforms)
    # # train_dataset = UCF101RetrievalDataset('../data/ucf101', 16, 10, True, train_transforms)    
    # train_dataloader = DataLoader(train_dataset, batch_size=8)

    # for i, data in enumerate(train_dataloader):
    #     clips, idxs = data
    #     # for i in range(10):
    #     #     filename = os.path.join('{}.mp4'.format(i))
    #     #     skvideo.io.vwrite(filename, clips[0][i])
    #     print(clips.shape)
    #     print(idxs)
    #     exit()
