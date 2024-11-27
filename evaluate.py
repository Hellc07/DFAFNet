import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.Baseline_Att import AttBaseline
from model.Baseline_Att_PIKD import AttPIKDBaseline
from model.Baseline_Att_AttUp import AttFAttUpBaseline
from model.Baseline_Att_PIKD_AttUp import AttFPIKDAttUpBaseline
from dataset import ValidDataSet
from utils import lovasz_losses as L
from utils.metrics import Evaluator
import os.path as osp

MODEL = 'AttFPIKDBaseline'
DATA_DIRECTORY = '/root/Datasets/US3D/test'
IGNORE_LABEL = 255
NUM_CLASSES = 5
INPUT_CHANNEL_RGB = 3
INPUT_CHANNEL_D =1
RESTORE_FROM = osp.join('/root/Module/AFAR_Net/snapshots/pretrained/US3D/', MODEL, 'best_model.pth.tar')
PRETRAINED_MODEL = None
SAVE_DIRECTORY = '/root/Module/Remote_ImageSS/results/US3D/Baseline_Res'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--input_ch_rgb", type=int, default=INPUT_CHANNEL_RGB,
                        help="number of input channels")
    parser.add_argument("--input_ch_d", type=int, default=INPUT_CHANNEL_D)
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--dataset", type=str, default="US3D",
                        help="Dataset's name")
    return parser.parse_args()


class IGRSSColorize(object):

    def __init__(self, args, n=10):

        if args.dataset == "SEN12MS":
            self.cmap = [(0, 153, 0),
                         (198, 176, 68),
                         (251, 255, 19),
                         (182, 255, 5),
                         (39, 255, 135),
                         (194, 79, 68),
                         (165, 165, 165),
                         (105, 255, 248),
                         (249, 255, 164),
                         (28, 13, 255)]
        if args.dataset == "US3D":
            self.cmap = [(255, 0, 0),
                         (204, 255, 0),
                         (0, 152, 102),
                         (0, 102, 255),
                         (204, 0, 255), ]

        self.cmap = np.array(self.cmap)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image


def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('Forest',  # always index 0
               'Shrubland', 'Savanna', 'Grassland', 'Wetlands',
               'Croplands', 'Urban', 'Snow_Ice', 'Barren', 'Water'))
    colormap = [(0, 0.6, 0),
                (0, 176/255.0, 68/255.0),
                (251/255.0, 1, 19/255.0),
                (182/255.0, 1, 5/255.0),
                (39/255.0, 1, 135/255.0),
                (39/255.0, 1, 135/255.0),
                (194/255.0, 79/255.0, 68/255.0),
                (165/255.0, 165/255.0, 165/255.0),
                (105/255.0, 1, 248/255.0),
                (249/255.0, 1, 164/255.0),
                (28/255.0, 13/255.0, 1)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)
    #ax1.imshow(gt)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    in_ch_rgb = args.input_ch_rgb
    in_ch_d = args.input_ch_d
    out_ch = args.num_classes

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.model == 'AttFPIKDAttUpBaseline':
        model = AttFPIKDAttUpBaseline(in_ch_rgb=in_ch_rgb, in_ch_d=in_ch_d, out_ch=out_ch)
    elif args.model == 'AttFAttUpBaseline':
        model = AttFAttUpBaseline(in_ch_rgb=in_ch_rgb, in_ch_d=in_ch_d, out_ch=out_ch)
    elif args.model == 'AttFBaseline':
        model = AttBaseline(in_ch_rgb=in_ch_rgb, in_ch_d=in_ch_d, out_ch=out_ch)
    else:
        model = AttPIKDBaseline(in_ch_rgb=in_ch_rgb, in_ch_d=in_ch_d, out_ch=out_ch)

    model = torch.nn.DataParallel(model)
    if args.restore_from[:4] == 'http':
        checkpoint = model_zoo.load_url(args.restore_from)
    else:
        checkpoint = torch.load(args.restore_from)
    model.load_state_dict(checkpoint['state_dict'])

    # print(checkpoint['epoch'])
    model.eval()
    model.cuda()

    testloader = data.DataLoader(ValidDataSet(args.data_dir, crop_size=(512, 512), scale=False, mirror=False),
                                 batch_size=15, shuffle=False, pin_memory=True)
    testdata_size = len(testloader)

    tbar = tqdm(testloader, desc='\r')
    evaluator = Evaluator(NUM_CLASSES)
    test_loss = 0.0
    # Visualize the test demo
    # visualizer = IGRSSColorize(args, n=NUM_CLASSES)

    for i, batch in enumerate(tbar):
        images, labels, depth, _, name = batch
        labels = labels.cuda()
        with torch.no_grad():
            if 'PIKD' in args.model:
                output, _ = model(images, depth)
                output = nn.functional.softmax(output)
            else:
                output = nn.functional.softmax(model(images, depth))

        loss = L.lovasz_softmax(output, labels)

        pred = output.data.cpu().numpy()
        target = labels.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        # colorful_pred = visualizer.__call__(pred.squeeze(axis=0))

        # imageio.imsave(os.path.join(args.save_dir, name[0] + ".png"),
        #                colorful_pred.transpose((1, 2, 0)))

        evaluator.add_batch(target, pred)
        test_loss += loss.item()
        tbar.set_description('Test loss: %.6f' % (test_loss / (i + 1)))


    PA = evaluator.Pixel_Accuracy()
    AA = evaluator.Pixel_Accuracy_Class()
    AA_List = evaluator.Pixel_Accuracy_list()
    Kappa = evaluator.Kappa()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (1, testdata_size))
    print("PA:{}, AA:{}, Kappa:{}, MIoU:{}, FWIoU: {}".format(PA, AA, Kappa, mIoU, FWIoU))
    for i in range(NUM_CLASSES):
        print("AA-class%d:"%(i+1), AA_List[i])
    evaluator.Show_Matrix()
    print('Loss: %.3f' % test_loss)


if __name__ == '__main__':
    main()
