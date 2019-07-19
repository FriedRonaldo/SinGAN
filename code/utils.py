import os
import torch
import numpy as np
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def formatted_print(notice, value):
    print('{0:<40}{1:<40}'.format(notice, value))


def save_checkpoint(state, check_list, log_dir, epoch=0):
    check_file = os.path.join(log_dir, 'model_{}.ckpt'.format(epoch))
    torch.save(state, check_file)
    check_list.write('model_{}.ckpt\n'.format(epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, prob):
    return prob * F.cross_entropy(pred, y_a) + (1 - prob) * F.cross_entropy(pred, y_b)


def mix_data(x, x_flip):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    # lam = np.random.uniform(0.0, 1.0)
    lam = np.random.beta(0.4, 0.4)

    x_mix = lam * x + (1 - lam) * x_flip
    return x_mix, lam


def get_att_map(feats, y_, norm=True):
    with torch.no_grad():
        y_ = y_.long()
        att_map = Variable(torch.zeros([feats.shape[0], feats.shape[2], feats.shape[3]]))

        for idx in range(feats.shape[0]):
            att_map[idx, :, :] = torch.squeeze(feats[idx, y_.data[idx], :, :])

        if norm:
            att_map = norm_att_map(att_map)

    return att_map


def norm_att_map(att_maps):
    _min = att_maps.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
    _max = att_maps.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    att_norm = (att_maps - _min) / (_max - _min)
    return att_norm


def load_bbox_size(dataset_path='../data/CUB/CUB_200_2011', img_size=224):
    origin_bbox = {}
    image_sizes = {}
    resized_bbox = {}
    with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])

            x, y, bbox_width, bbox_height = map(float, file_info[1:])

            origin_bbox[image_id] = [x, y, bbox_width, bbox_height]

    with open(os.path.join(dataset_path, 'sizes.txt')) as f:
        for each_line in f:
            file_info = each_line.strip().split()
            image_id = int(file_info[0])
            image_width, image_height = map(float, file_info[1:])

            image_sizes[image_id] = [image_width, image_height]

    for i in origin_bbox.keys():
        x, y, bbox_width, bbox_height = origin_bbox[i]
        image_width, image_height = image_sizes[i]

        x_scale = img_size / image_width
        y_scale = img_size / image_height

        x_new = int(np.round(x * x_scale))
        y_new = int(np.round(y * y_scale))
        x_max = int(np.round(bbox_width * x_scale))
        y_max = int(np.round(bbox_height * y_scale))

        resized_bbox[i] = [x_new, y_new, x_new + x_max, y_new + y_max]

    return resized_bbox


def cammed_image(image, mask, require_norm=False):
    if require_norm:
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return heatmap * 255., cam * 255.


def intensity_to_rgb(intensity, normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.
    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.
    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def large_rect(rect):
    # find largest recteangles
    large_area = 0
    target = 0
    if len(rect) == 1:
        x = rect[0][0]
        y = rect[0][1]
        w = rect[0][2]
        h = rect[0][3]
        return x, y, w, h
    else:
        for i in range(1, len(rect)):
            area = rect[i][2] * rect[i][3]
            if large_area < area:
                large_area = area
                target = i
        x = rect[target][0]
        y = rect[target][1]
        w = rect[target][2]
        h = rect[target][3]
        return x, y, w, h


def calculate_IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def calculate_IOU_multibox(boxA, boxesB):

    interArea = 0.0
    boxAArea = 0.0
    boxBArea = 0.0

    for boxB in boxesB:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea += max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxBArea += (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea += (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class EMA(torch.nn.Module):
    def __init__(self, mu=0.999):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def cross_entropy(input, target):
    """ Cross entropy for one-hot labels
    """
    return -torch.mean(torch.sum(target * F.log_softmax(input), dim=1))


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, args):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


# my Rectangle = (x1, y1, x2, y2), a bit different from OP's x, y, w, h
def intersection(rectA, rectB): # check if rect A & B intersect
    a, b = rectA, rectB
    startX = max(min(a[0], a[2]), min(b[0], b[2]))
    startY = max(min(a[1], a[3]), min(b[1], b[3]))
    endX = min(max(a[0], a[2]), max(b[0], b[2]))
    endY = min(max(a[1], a[3]), max(b[1], b[3]))
    if startX < endX and startY < endY:
        return True
    else:
        return False


def combineRect(rectA, rectB): # create bounding box for rect A & B
    a, b = rectA, rectB
    startX = min(a[0], b[0])
    startY = min(a[1], b[1])
    endX = max(a[2], b[2])
    endY = max(a[3], b[3])
    return (startX, startY, endX, endY)


def checkIntersectAndCombine(rects):
    if rects is None:
        return None
    mainRects = rects
    noIntersect = False
    while noIntersect == False and len(mainRects) > 1:
        # mainRects = list(set(mainRects))
        # get the unique list of rect, or the noIntersect will be
        # always true if there are same rect in mainRects
        newRectsArray = []
        for rectA, rectB in itertools.combinations(mainRects, 2):
            newRect = []
            if intersection(rectA, rectB):
                newRect = combineRect(rectA, rectB)
                newRectsArray.append(newRect)
                noIntersect = False
                # delete the used rect from mainRects
                if rectA in mainRects:
                    mainRects.remove(rectA)
                if rectB in mainRects:
                    mainRects.remove(rectB)
        if len(newRectsArray) == 0:
            # if no newRect is created = no rect in mainRect intersect
            noIntersect = True
        else:
            # loop again the combined rect and those remaining rect in mainRects
            mainRects = mainRects + newRectsArray
    return mainRects

