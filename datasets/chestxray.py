import csv
import os
import warnings

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


class ChestXray(VisionDataset):
    class_names_map = {
        'No Finding': 0, 'Atelectasis': 1, 'Cardiomegaly': 2, 'Effusion': 3, 'Infiltration': 4,
        'Mass': 5, 'Nodule': 6, 'Pneumonia': 7, 'Pneumothorax': 8, 'Consolidation': 9,
        'Edema': 10, 'Emphysema': 11, 'Fibrosis': 12, 'Pleural_Thickening': 13, 'Hernia': 14
    }
    prefix = 'images'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(ChestXray, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.loader = default_loader

        items = []

        if download is True:
            raise NotImplementedError
        warnings.warn('Multi-label samples will be skipped.')

        data_list_names = 'train_val_list.txt' if train else 'test_list.txt'

        with open(os.path.join(root, data_list_names), 'r') as f:
            data_list = {line.strip() for line in f}

        with open(os.path.join(root, 'Data_Entry_2017.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if '|' in row[1] or row[0] not in data_list:
                    continue
                items.append((os.path.join(self.root, self.prefix, row[0]), self.class_names_map[row[1]]))
        self.samples = items

    def __getitem__(self, index):
        path, target = self.samples[index]

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.samples)

    def get_class_count(self):
        counts = {}
        for index in range(len(self)):
            target = self.samples[index][1]
            if target in counts:
                counts[target] += 1
            else:
                counts[target] = 1
        ret = [counts[i] for i in range(len(counts))]
        return ret
