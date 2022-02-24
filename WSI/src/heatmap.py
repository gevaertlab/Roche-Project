import os
import argparse
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from sklearn import preprocessing
from openslide import OpenSlide
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.exposure.exposure import is_low_contrast
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import matplotlib
matplotlib.use('Agg')

from wsi_model import *
from read_data import *
from resnet import resnet50

# mask functions
def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level


def assig_to_heatmap(heatmap, patch, x, y, ratio_patch_x, ratio_patch_y,xmax, ymax):
    new_x = int(x / ratio_patch_x)
    new_y = int(y / ratio_patch_y)
    
    try:
        if new_x+patch.shape[0] > heatmap.shape[0] and new_y+patch.shape[1] < heatmap.shape[1]:
            dif = new_x+patch.shape[0] - xmax
            dif = patch.shape[0] - dif
            heatmap[new_x:, new_y:new_y+patch.shape[1], :] = patch[:dif, :, :]
        elif new_x+patch.shape[0] < heatmap.shape[0] and new_y+patch.shape[1] > heatmap.shape[1]:
            dif = new_y+patch.shape[1] - ymax
            dif = patch.shape[1] - dif
            heatmap[new_x:new_x+patch.shape[0], new_y:, :] = patch[:, :dif, :]
        elif new_x+patch.shape[0] > heatmap.shape[0] and new_y+patch.shape[1] > heatmap.shape[1]:
            return heatmap
        else:
            heatmap[new_x:new_x+patch.shape[0], new_y:new_y+patch.shape[1], :] = patch
    
        return heatmap
    except:
        return heatmap

def get_prob_reg(output, thresholds):
    """Compute a class based on a regression prediction

    Args:
        output (torch.tensor): probabilities returned by the model
        thresholds (List): List with the thresholds that need to be used to compute the class
    """
    n_ranges = len(thresholds)
    probs = np.zeros(n_ranges+1)
    over_range = True
    for i in range(0, n_ranges, 2):
        if output > i and output <= i+1:
            if output > thresholds[i]:
                probs[i+1] = 1
                pos = i+1
            else:
                probs[i] = 1
                pos = i
            over_range = False
            break
    if over_range: 
        probs[-1] = 1
        pos = n_ranges
    
    return probs, pos

def generate_heatpmap(slide_path: str, patch_size: Tuple, slide_id: str, model: nn.Module,
                      args: dict):
    model.eval()
    transforms_ = torch.nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    slide = OpenSlide(slide_path)

    target_layers = [model.resnet.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.cuda)

    mask, mask_level = get_mask(slide)
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=3)

    mask_level = len(slide.level_dimensions) - 1
    
    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
    ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

    xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

    # handle slides with 40 magnification at base level
    resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    resize_factor = resize_factor * args.dezoom_factor
    resize_factor = 1.0
    patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
    i = 0

    indices = [(x, y) for x in range(0, xmax, 256) for y in
                range(0, ymax, 256)]
    #np.random.seed(5)
    #np.random.shuffle(indices)
    # create array to insert the patches generated
    xmax_patch = int(xmax/16)
    ymax_patch = int(xmax/16)

    ratio_patch_x = int(xmax/xmax_patch)
    ratio_patch_y = int(ymax/ymax_patch)
    heatmap = np.zeros((xmax_patch, ymax_patch, 3))
    slide_resized = np.zeros((xmax_patch, ymax_patch, 3))
    colors = []

    if not args.grad_cam:
        import random
        random.seed(99)
        for i in range(len(args.classes)):
            hex = '#%06X' % random.randint(0, 0xFFFFFF)
            value = hex.lstrip('#')
            lv = len(value)
            rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
            bgr = (rgb[0], rgb[1], rgb[2])
            colors.append(rgb)
        areas_class = dict()
        class_predictions = dict()
        for label,color in zip(args.classes, colors):
            print('{}/{}'.format(label,color))
            areas_class[label] = 0
            class_predictions[label] = 0
        
        # variables for statistics
        probabilities = []
        tissue_area = 0
        num_total_tiles = 0

    for x, y in tqdm(indices):
        # check if in background mask
        x_mask = int(x / ratio_x)
        y_mask = int(y / ratio_y)
        patch = slide.read_region((x, y), PATCH_LEVEL, (256,256)).convert('RGB')
        if mask[x_mask, y_mask] == 1:
            try:
                mask_patch = get_mask_image(np.array(patch))
                mask_patch = binary_dilation(mask_patch, iterations=3)
            except Exception as e:
                print("error with slide id {} patch {}".format(slide_id, i))
                print(e)
            if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                if resize_factor != 1.0:
                    patch = patch.resize(patch_size)
                bagpatch = torch.from_numpy(np.array(patch)).permute(2,0,1)
                bagpatch = bagpatch.unsqueeze(0).unsqueeze(0) # necessary for being a bag of patches
                if args.cuda:
                    bagpatch = bagpatch.to('cuda:0')
                bagpatch = transforms_(bagpatch)
                if args.grad_cam:
                    targets = [ClassifierOutputTarget(args.category[0])]
                    label = torch.tensor(args.category, dtype=torch.float32)
                    grayscale_cam = cam(input_tensor=bagpatch, targets=targets)
                    #grayscale_cam = grayscale_cam[0, :]
                    c, h, w = grayscale_cam.shape
                    grayscale_cam = grayscale_cam.reshape(h,w,c)
                    patch = np.array(patch)/255
                    visualization = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
                    
                    visualization = cv2.resize(visualization, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
                else:
                    with torch.set_grad_enabled(False):
                        outputs = model(bagpatch)
                    if args.regression:
                        probs, pos = get_prob_reg(outputs, args.thresholds)
                        probabilities.append(probs)
                    else:
                        probs = F.softmax(outputs.detach().cpu(), dim=1)
                        probabilities.append(probs.numpy())
                        _, pos = torch.max(probs, 1)
                    color = colors[pos]
                    visualization = np.empty((64,64,3), np.uint8)
                    visualization[:] = color[0], color[1], color[2]
                    tissue_area += 64*64
                    areas_class[args.classes[pos]] += 64*64
                    class_predictions[args.classes[pos]] += 1
                    num_total_tiles += 1

                heatmap = assig_to_heatmap(heatmap,visualization, x, y, 
                                           ratio_patch_x, ratio_patch_y, xmax_patch, ymax_patch)
                patch = patch.resize((64,64))
                patch = np.array(patch)
                slide_resized = assig_to_heatmap(slide_resized,patch, x, y, 
                                           ratio_patch_x, ratio_patch_y, xmax_patch, ymax_patch)
            else:
                #if resize_factor != 1.0:
                patch = patch.resize((64,64))
                patch = np.array(patch)
               
                try:
                    heatmap = assig_to_heatmap(heatmap,patch, x, y, 
                                               ratio_patch_x, ratio_patch_y, xmax_patch, ymax_patch)
                    slide_resized = assig_to_heatmap(slide_resized,patch, x, y, 
                                           ratio_patch_x, ratio_patch_y, xmax_patch, ymax_patch)
                except: 
                    continue
        else:
            patch = patch.resize((64,64))
            patch = np.array(patch)
            try:
                heatmap = assig_to_heatmap(heatmap, patch, x, y, 
                                           ratio_patch_x, ratio_patch_y, xmax_patch, ymax_patch)
                slide_resized = assig_to_heatmap(slide_resized,patch, x, y, 
                                           ratio_patch_x, ratio_patch_y, xmax_patch, ymax_patch)
            except:
                continue
    
    statistics = {
        'probabilities': np.array(probabilities)
    }
    if not args.grad_cam:
        statistics['colors'] = colors
        statistics['tissue_area'] = tissue_area
        statistics['area_class'] = areas_class
        statistics['num_total_tiles'] = num_total_tiles
        statistics['class_predictions'] = class_predictions

    return heatmap, slide_resized, statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heatmap generation')
    parser.add_argument('--wsi_path', required=True, metavar='WSI_PATH', type=str,
                    help='Path to the input directory of WSI files')
    parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                                                                    'default 768')
    parser.add_argument('--dezoom_factor', default=1.0, type=float,
                        help='dezoom  factor, 1.0 means the images are taken at 20x \
                              magnification, 2.0 means the images are taken at 10x magnification')
    parser.add_argument('--checkpoint', type=str, help='checkpoint for the image model.')
    parser.add_argument('--cuda', default=True, help='checkpoint for the image model.')
    parser.add_argument('--problem', default='tissue', help='problem type to load classes.')
    parser.add_argument('--grad_cam', default=0, type=int)
    parser.add_argument('--label', help='The class of the sample.')
    parser.add_argument('--output_dir', default=None, help='Output dir for the image.')
    parser.add_argument('--suffix', default=None, help='Suffix for the image to be saved.')
    parser.add_argument('--regression', action="store_true", help='If we are treating the problem as regression.')
    parser.add_argument('--thresholds', default=None, nargs='+', help='Thresholds to be used if regression wanted to be used.')
    args = parser.parse_args()

    if args.problem == 'tissue':
        args.classes = ['Brain', 'Eshopagus', 'Eyes', 'Kidney', 'Lung', 'Ovary', 'Pancreas', 'Rectum', 'Uterus']
    elif args.problem == 'stage':
        args.classes = ['StageI', 'StageII', 'StageIII', 'StageIV']
    elif args.problem == 'grade':
        args.classes = ['Pattern3', 'Pattern4', 'Pattern5']
    elif args.problem == 'gleason':
        args.classes = ['Pattern10', 'Pattern6', 'Pattern7', 'Pattern8', 'Pattern9']

    if args.regression:
        args.num_outputs = 1
        if not args.thresholds:
            args.thresholds = [x+0.5 for x in range(len(args.classes)-1)]
    else:
        args.num_outputs = len(args.classes)
    le = preprocessing.LabelEncoder()
    le.fit(args.classes)
    args.category = le.transform(np.array([args.label]))
    resnet50 = resnet50(pretrained=True)

    layers_to_train = [resnet50.fc, resnet50.layer4, resnet50.layer3]
    for param in resnet50.parameters():
        param.requires_grad = False
    for layer in layers_to_train:
        for n, param in layer.named_parameters():
            param.requires_grad = True

    model = AggregationModel(resnet50, num_outputs=args.num_outputs)

    if args.checkpoint is not None:
        print('Restoring from checkpoint')
        print(args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint))

    model = model.to('cuda:0')
    args.patch_size = [args.patch_size, args.patch_size]
    heatmap, slide_resized, statistics = generate_heatpmap(args.wsi_path, args.patch_size, 0, model, args)
    heatmap = heatmap.astype(np.uint8)
    slide_resized = slide_resized.astype(np.uint8)
    name_save = args.wsi_path.split('/')[-1].replace('.svs','')
    
    if args.suffix:
        name_image = f'{name_save}-{args.suffix}-heatmap.png'
        name_statistics = f'{name_save}-{args.suffix}-statistics.txt'
        name_histo = f'{name_save}-{args.suffix}-histo.png'
    else:
        name_image = f'{name_save}-heatmap.png'
        name_statistics = f'{name_save}-statistics.txt'
        name_histo = f'{name_save}-histo.png'
    
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        name_image = os.path.join(args.output_dir, name_image)
        name_statistics = os.path.join(args.output_dir, name_statistics)
        name_histo = os.path.join(args.output_dir, name_histo)

    if not args.grad_cam:
        # save images
        f, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
        colors = statistics['colors']
        ax1.imshow(heatmap)
        ax2.imshow(slide_resized)
        #ax = plt.gca()
        #  hide x-axis
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        handles = [Rectangle((0,0),1,1, color =tuple(map(lambda x: x/255,c))) for c in colors]
        plt.title(args.label)
        plt.legend(handles, args.classes,loc='upper center', 
                bbox_to_anchor=(1.15, 1),fancybox=False, shadow=False)
        plt.tight_layout()
        plt.savefig(name_image, dpi=300)
        plt.close()

        # save statistics on file
        with open(name_statistics, 'w') as f:
            f.write('Colors \n')
            for key, value in zip(args.classes,statistics['colors']):
                f.write(f'  {key}: {value}\n')

            f.write('Tissue area/tiles per class: \n')
            for key, value in statistics['area_class'].items():
                class_area = (value/statistics['tissue_area'])*100
                tiles = statistics['class_predictions'][key]
                f.write(f'  {key}: {class_area} % | N tiles {tiles}\n')
        
        # save histogram of probabilities
        probs = statistics['probabilities'].reshape(statistics['probabilities'].shape[0], len(args.classes))
        fig, axs = plt.subplots(1, len(args.classes), sharey=True, tight_layout=True)
        for i in range(len(args.classes)):
            axs[i].hist(probs[:,i], bins=20)
            axs[i].set_title(args.classes[i])
        plt.savefig(name_histo, dpi=300)
    else:
        concat_image = cv2.hconcat([slide_resized, heatmap])
        cv2.imwrite(name_image, concat_image)

