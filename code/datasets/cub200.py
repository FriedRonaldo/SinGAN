import os
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as vF
from PIL import Image
import numpy as np


def pil_loader(path, gray=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if gray:
            return img.convert('L')
        else:
            return img.convert('RGB')


# class Cub2011(Dataset):
#     base_folder = 'CUB_200_2011/images'
#     url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
#     filename = 'CUB_200_2011.tgz'
#     tgz_md5 = '97eceeb196236b17998738112f37df78'
#
#     def __init__(self, root, train=True, transform=None, download=True, with_id=False):
#         self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.loader = default_loader
#         self.train = train
#         self.with_id = with_id
#
#         if download:
#             self._download()
#
#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')
#
#     def _load_metadata(self):
#         images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
#                              names=['img_id', 'filepath'])
#         image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
#                                          sep=' ', names=['img_id', 'target'])
#         train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
#                                        sep=' ', names=['img_id', 'is_training_img'])
#
#         data = images.merge(image_class_labels, on='img_id')
#         self.data = data.merge(train_test_split, on='img_id')
#
#         if self.train:
#             self.data = self.data[self.data.is_training_img == 1]
#         else:
#             self.data = self.data[self.data.is_training_img == 0]
#
#     def _check_integrity(self):
#         try:
#             self._load_metadata()
#         except Exception:
#             return False
#
#         for index, row in self.data.iterrows():
#             filepath = os.path.join(self.root, self.base_folder, row.filepath)
#             if not os.path.isfile(filepath):
#                 print(filepath)
#                 return False
#         return True
#
#     def _download(self):
#         import tarfile
#
#         if self._check_integrity():
#             print('Files already downloaded and verified')
#             return
#
#         download_url(self.url, self.root, self.filename, self.tgz_md5)
#
#         with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
#             tar.extractall(path=self.root)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         sample = self.data.iloc[idx]
#         path = os.path.join(self.root, self.base_folder, sample.filepath)
#         target = sample.target - 1  # Targets start at 1 by default, so shift to 0
#         img_id = sample.img_id
#         img = self.loader(path)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.with_id:
#             return img, target, img_id
#         else:
#             return img, target


class Cub2011Rot(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, download=True, with_id=False, num_tf=3):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.with_id = with_id
        self.num_tf = num_tf

        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        # target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img_id = sample.img_id
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        img = self.normalize(img)

        rot = np.random.choice(4)

        img_rot = img.rot90(rot, (1, 2))

        if self.with_id:
            return img, img_rot, rot, img_id
        else:
            return img, img_rot, rot


class Cub2011transform(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, download=True, with_id=False, num_tf=3):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.with_id = with_id
        self.num_tf = num_tf

        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img_id = sample.img_id
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        imgtf = img.copy()

        use_tf = np.random.binomial(1, 0.5, self.num_tf)

        signs = np.random.choice([1, -1], size=3)

        rot = signs[0] * np.random.randint(30, 180)
        translate = signs[1] * np.random.randint(10, 60)
        shear = signs[2] * np.random.randint(10, 30)

        use_tf[1] = 0

        if use_tf[0]:
            imgtf = vF.affine(imgtf, rot, (0, 0), 1.0, 0, 0)
        if use_tf[1]:
            imgtf = vF.affine(imgtf, 0, (translate, translate), 1.0, 0, 0)
        if use_tf[2]:
            imgtf = vF.affine(imgtf, 0, (0, 0), 1.0, shear, 0)

        img = self.normalize(img)
        imgtf = self.normalize(imgtf)

        if self.with_id:
            return img, imgtf, target, use_tf, torch.Tensor([rot, translate, shear] * use_tf), img_id
        else:
            return img, imgtf, target, use_tf, torch.Tensor([rot, translate, shear] * use_tf)


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, label=True, transform=None, download=True, with_id=False, n_labels=2000, mode='semi'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.label = label
        self.with_id = with_id
        self.n_labels = n_labels
        self.mode = mode

        if download:
            self._download()

        if self.train:
            self._train_split(n_labels)
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _train_split(self, n_labels):

        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        train_list = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_list.txt'), sep=';',
                                 names=['filepath', 'target'])
        data = images.merge(train_list, on='filepath')

        labeled_train_list = []
        unlabeled_train_list = []
        for i in range(200):
            cur_data = data.loc[(data['target'] == i), :]
            cur_labeled_train_data = cur_data.iloc[:n_labels // 200, :]
            cur_unlabeled_train_data = cur_data.iloc[n_labels // 200:, :]
            labeled_train_list.append(cur_labeled_train_data)
            unlabeled_train_list.append(cur_unlabeled_train_data)

        labeled_df = pd.concat(labeled_train_list)
        unlabeled_df = pd.concat(unlabeled_train_list)

        labeled_df.to_csv(os.path.join(self.root, 'CUB_200_2011', 'labeled_train_list.txt'),
                          sep=' ',
                          header=False,
                          index=False)
        unlabeled_df.to_csv(os.path.join(self.root, 'CUB_200_2011', 'unlabeled_train_list.txt'),
                            sep=' ',
                            header=False,
                            index=False)

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train and self.label and self.mode == 'semi':
            self.data = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'labeled_train_list.txt'),
                                    sep=' ',
                                    names=['img_id', 'filepath', 'target'])
        elif self.train and self.mode == 'semi':
            self.data = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'unlabeled_train_list.txt'),
                                    sep=' ',
                                    names=['img_id', 'filepath', 'target'])
        elif self.train and self.mode == 'full':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        if self.train and self.mode == 'semi':
            target = sample.target
        else:
            target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img_id = sample.img_id
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.with_id:
            return img, target, img_id
        else:
            return img, target


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def get_cub200(root, n_labeled=2000, transform_train=None, transform_val=None, mode='semi'):
    train_labeled_dataset = Cub2011(root, train=True, label=True, transform=transform_train, n_labels=n_labeled, mode=mode)
    if mode == 'semi':
        train_unlabeled_dataset = Cub2011(root, train=True, label=False, transform=TransformTwice(transform_train), n_labels=n_labeled, mode=mode)
    else:
        train_unlabeled_dataset = []
    val_dataset = Cub2011(root, train=False, label=False, transform=transform_val, with_id=True, mode=mode)

    print("#Labeled: {} #Unlabeled: {} #Val: {}".format(len(train_labeled_dataset), len(train_unlabeled_dataset), len(val_dataset)))
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset
