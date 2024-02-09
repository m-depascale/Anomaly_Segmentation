# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from enet import ENet
from erfnet import ERFNet
from bisenetv1 import BiSeNetV1
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torch.nn.functional as F
from torchvision.transforms import Resize


seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', type=str, default='MSP')
    parser.add_argument('--temp', type=float, default=0.)
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    ### FOR VOID CLASSIFIER WE ALSO USE ENET AND BISENET 
    if args.loadModel == 'erfnet.py':
        model = ERFNet(NUM_CLASSES)
    elif args.loadModel == 'enet.py':
        model = ENet(NUM_CLASSES)
    elif args.loadModel == 'bisenetv1.py':
        model = BiSeNetV1(NUM_CLASSES, aux_mode='eval')    

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    if args.loadModel == 'erfnet.py':
      model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    
    elif args.loadModel == 'enet.py':
      print('path w', weightspath)
      state_dict = torch.load(weightspath)['state_dict']
      # Remove 'module.' prefix from keys if present
      new_dict = {}
      for key, value in state_dict.items():
        new_dict['module.'+key] = value
      model.load_state_dict(new_dict)

    elif args.loadModel == 'bisenetv1.py': #bisenetv1.py
      state_dict = torch.load(weightspath)
        # Remove 'module.' prefix from keys if present
      new_dict = {}
      print("--- SAVED PRETRAINED PARAMS ---- ")
      for key, value in state_dict.items():
        #print(key)
        if key.split('.')[0] not in ['conv_out16', 'conv_out32']:
          new_dict['module.'+key] = value
        #print('module.'+key)
      model.load_state_dict(new_dict)

      
    #print('model: ', model)
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        #print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        #print(args.loadModel)
        if args.loadModel == 'bisenetv1.py':
            images = Resize((1024,2048), Image.BILINEAR)(images)
        with torch.no_grad():
          if args.loadModel == 'bisenetv1.py':
            #print(images.shape)
            result = model(images)[0]
            #print(result.shape)
          else:
            result = model(images)
        
        if args.method == 'VOID':    
          anomaly_result = F.softmax(result.squeeze(0), dim=0).data.cpu().numpy()[-1]

        if args.method == 'MSP':
            anomaly_result = 1.0 - np.max(F.softmax(result.squeeze(0), dim=0).data.cpu().numpy(), axis=0)            
        elif args.method == 'MSPT':
            #MSP with temp scaling
            temp = args.temp if args.temp>0 else 1
            print('temp: ', temp)
            anomaly_result = 1.0 - np.max(F.softmax(result.squeeze(0)/temp, dim=0).data.cpu().numpy(), axis=0)                        
        elif args.method == 'MaxLogit':
            anomaly_result =  1 - np.max(F.normalize(result.squeeze(0), dim=0).data.cpu().numpy(), axis=0)
        elif args.method == 'MaxEntropy':
            anomaly_result = torch.div(torch.sum(-F.softmax(result.squeeze(0), dim=0) * F.log_softmax(result.squeeze(0), dim=0), dim=0),
                                       torch.log(torch.tensor(F.softmax(result.squeeze(0), dim=0).size(0)))).data.cpu().numpy()
            
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        if args.loadModel == 'bisenetv1.py':
            mask = Resize((1024, 2048), Image.NEAREST)(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    #print(len(ood_gts), len(anomaly_scores))

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)
    #print(len(ood_mask), len(ind_mask))

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    #print(len(ood_out), len(ind_out))

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    #print(len(ood_label), len(ind_label))
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    #print(len(val_out), len(val_label))
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')
    if args.temp == 0.:
        file.write(f'METHOD: {args.method} DATASET: {args.input} -- AUPRC score: {prc_auc*100.0}, FPR@TPR95: {fpr*100.0}') 
    else: 
        file.write(f'METHOD: {args.method} TEMP {args.temp} DATASET: {args.input} -- AUPRC score: {prc_auc*100.0}, FPR@TPR95: {fpr*100.0}') 

    file.close()

if __name__ == '__main__':
    main()
