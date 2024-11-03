import os
import argparse
import math
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from gl_mae_model import *
from mydataset import MyDataset
from utils import setup_seed


def main(args):

    batch_size = args.batch_size

    test_dataset = MyDataset(args.root2, args.txtpath2, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)

    args.model_path = 'vit-t-classifier-from_scratch_gl.pt'

    if args.model_path is not None:
        model = torch.load(args.model_path).to(device)
    else:
        raise ValueError("model path is None")

    # writer = SummaryWriter(os.path.join('logs', 'mae', 'cls_results'))
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    model.eval()
    with torch.no_grad():
        losses = []
        acces = []
        # for img, label in tqdm(iter(val_dataloader)):
        for (cnt, i) in enumerate(test_loader):
            # print("train-cnt:", cnt)
            # if(cnt>2):break
            batch_x = i['data']
            batch_y = i['label']
            batch_x = torch.unsqueeze(batch_x, dim=1)
            batch_x = batch_x.float()
            if torch.cuda.is_available():
                #     print("torch:",torch.cuda.is_available())
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
            batch_x = batch_x.reshape(batch_x.shape[0], 100, 100, batch_x.shape[3])
            # 调整维度顺序
            batch_x = batch_x.permute(0, 3, 1, 2)
            img = batch_x
            label = batch_y

            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            losses.append(loss.item())
            acces.append(acc.item())
        avg_test_loss = sum(losses) / len(losses)
        avg_test_acc = sum(acces) / len(acces)
        print(f'average test loss is {avg_test_loss}, average test acc is {avg_test_acc}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--model_path', type=str, default=None)

    rootpath = 'das_data'
    parser.add_argument("--root", type=str, default=rootpath + '/train',
                        help="rootpath of traindata")
    parser.add_argument("--root2", type=str, default=rootpath + '/test',
                        help="rootpath of testdata")
    parser.add_argument("--root3", type=str, default=rootpath + '/val',
                        help="rootpath of valdata")

    parser.add_argument("--txtpath", type=str, default=rootpath + '/train/label.txt',
                        help="path of train_list")
    parser.add_argument("--txtpath2", type=str, default=rootpath + '/test/label.txt',
                        help="path of test_list")
    parser.add_argument("--txtpath3", type=str, default=rootpath + '/val/label.txt',
                        help="pach of val_list")

    args = parser.parse_args()
    main(args)

#none
'''
average test loss is 0.05239098165506763, average test acc is 0.9878826530612245.
'''

'''
average test loss is 0.02263434821525038, average test acc is 0.9974489795918368.
'''





