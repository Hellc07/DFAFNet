import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import pickle
from tqdm import tqdm
import timeit
from utils import lovasz_losses as L
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from model.Baseline_Att_PIKD_AttUp import AttFPIKDAttUpBaseline
from dataset import TrainDataSet
from utils.saver import Saver
from utils.summaries import TensorboardSummary

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 15
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/private/AFAR_Net/dataset/US3D/'
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 5
NUM_EPOCHS = 150
POWER = 0.9
RANDOM_SEED = 1234
SAVE_PRED_EVERY = 10
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005


def get_hidden_loss(s, t, e=1e-6):
    s_mean = s.mean(dim=1)
    t_mean = t.mean(dim=1)
    loss = torch.mean(torch.sqrt((s_mean - t_mean) ** 2 + e ** 2))
    return loss


def PIKDLoss(h, h_t):
    out = []
    for s, t in zip(h, h_t):
        out.append(get_hidden_loss(s, t))
    return torch.mean(torch.tensor(out))


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default="DeepLab",
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default="US3D",
                        help="The name of dataset")
    parser.add_argument("--checkname", type=str, default="Baseline",
                        help="THe name of your model")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--lr_scheduler", type=str, default="step", choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument("--valid_step", type=int, default=1,
                        help="The steps of validation")
    return parser.parse_args()


args = get_arguments()


def main():
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    cudnn.enabled = True

    model = AttFPIKDAttUpBaseline(in_ch_rgb=3, in_ch_d=1, out_ch=args.num_classes)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    model.cuda()

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")

    train_dataset = TrainDataSet(train_dir, crop_size=input_size, scale=False, mirror=False, mean=IMG_MEAN)
    valid_dataset = TrainDataSet(valid_dir, crop_size=input_size, scale=False, mirror=False, mean=IMG_MEAN)
    train_dataset_size = len(train_dataset)
    valid_dataset_size = len(valid_dataset)

    train_ids = np.array(list(range(train_dataset_size)))
    valid_ids = np.array(list(range(valid_dataset_size)))

    np.random.shuffle(train_ids)
    np.random.shuffle(valid_ids)

    pickle.dump(train_ids, open(osp.join(args.snapshot_dir, 'train_id.pkl'), 'wb'))
    pickle.dump(valid_ids, open(osp.join(args.snapshot_dir, 'valid_id.pkl'), 'wb'))

    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=3, pin_memory=True)
    validloader = data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=3, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    optimizer.zero_grad()
    evaluator = Evaluator(args.num_classes)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.num_epochs, train_dataset_size, lr_step=100)
    best_pred = 0.0
    saver = Saver(args)
    saver.save_experiment_config()
    summary = TensorboardSummary(saver.directory)
    writer = summary.create_summary()

    for epoch in range(args.num_epochs):
        train_loss = 0.0
        fuse_loss = 0.0
        model.train()
        tbar = tqdm(trainloader)

        for i, batch in enumerate(tbar):

            images, labels, depth, _, name = batch
            scheduler(optimizer, i, epoch, best_pred=best_pred)
            optimizer.zero_grad()

            images = Variable(images).cuda()
            depth = Variable(depth).cuda()
            depth = depth.unsqueeze(1)
            labels = labels.cuda()

            if args.checkname == "ResBranch3D_Edge":
                output, d1, d2, d3, d4, d5, fuse = model(images)
                output = nn.functional.softmax(output)
            else:
                output, feature = model(images, depth)
                output = nn.functional.softmax(output)

            loss_seg = L.lovasz_softmax(output, labels)

            loss_fuse = PIKDLoss(feature[0], feature[1]) + PIKDLoss(feature[0], feature[2])
            loss = loss_seg + loss_fuse

            loss.backward()
            optimizer.step()

            train_loss += loss_seg
            fuse_loss += loss_fuse
            tbar.set_description('Train loss: %.6f, Fuse loss: %.6f' % (train_loss/(i+1), fuse_loss/(i+1)))
            writer.add_scalar('train/total_loss_iter', train_loss, i + train_dataset_size * epoch)

        writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, train_dataset_size))
        print('Loss: %.3f' % train_loss)

        if (epoch + 1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.dataset, args.checkname,
                                                    args.dataset+str(epoch)+'.pth.tar'))

        # save checkpoint every epoch
        is_best = False
        saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            }, is_best, filename='new_model.pth.tar')

        torch.cuda.empty_cache()

        if (epoch + 1) % args.valid_step == 0:
            model.eval()
            evaluator.reset()
            tbar = tqdm(validloader, desc='\r')
            test_loss = 0.0
            for i, batch in enumerate(tbar):
                images, labels, depth, _, _ = batch
                images = Variable(images).cuda()
                depth = Variable(depth).cuda()
                depth = depth.unsqueeze(1)
                labels = labels.cuda()

                with torch.no_grad():
                    output, feature = model(images, depth)
                    output = nn.functional.softmax(output)

                loss = L.lovasz_softmax(output, labels)

                pred = output.data.cpu().numpy()
                target = labels.cpu().numpy()
                pred = np.argmax(pred, axis=1)

                evaluator.add_batch(target, pred)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.6f' % (test_loss/(i+1)))

            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            mIoU = evaluator.Mean_Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
            Kappa = evaluator.Kappa()

            writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
            writer.add_scalar('val/mIoU', mIoU, epoch)
            writer.add_scalar('val/Acc', Acc, epoch)
            writer.add_scalar('val/Acc_class', Acc_class, epoch)
            writer.add_scalar('val/fwIoU', FWIoU, epoch)
            writer.add_scalar('val/Kappa', Kappa, epoch)
            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, valid_dataset_size))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Kappa: {}".format(Acc, Acc_class, mIoU, FWIoU, Kappa))
            print('Loss: %.3f' % test_loss)

            new_pred = mIoU
            if new_pred > best_pred:
                is_best = True
                best_pred = new_pred
                saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_pred,
                }, is_best, filename='best_model.pth.tar')
        torch.cuda.empty_cache()

    end = timeit.default_timer()
    print(end-start, 'seconds')

if __name__ == '__main__':
    main()
