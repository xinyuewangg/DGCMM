from functools import partial
# from torchvision.datasets import SVHN

from datasets.chestxray import ChestXray

# from datasets.imagenet_lt import ImageNet_LT
# from datasets.inat2017 import INat2017
# from datasets.myimagenet import MyImageNet
# from datasets.places_lt import Places_LT

_dataset_class_map = {
    # 'ImageNet': MyImageNet,
    # 'CIFAR10': CIFAR10,
    # 'CIFAR100': CIFAR100,
    # 'SVHN': SVHN,
    # 'iNaturalist2017': INat2017,
    # 'Places_LT': Places_LT,
    # 'ImageNet_LT': ImageNet_LT,
    'ChestXray': ChestXray

}
_dataset_class_number_map = {
    'ImageNet': 1000,
    'CIFAR10': 10,
    'CIFAR100': 100,
    'SVHN': 10,
    'iNaturalist2017': 13,
    'Places_LT': 365,
    'ImageNet_LT': 1000,
    'ChestXray': 15
}


# data_root_map = {
#     'Places_LT': '/data/public/datasets/Places365',
#     'ImageNet_LT': '/data/public/datasets/ImageNet'
# }
#

def get_dataset(dataset_name, **kwargs):
    if dataset_name in ('ImageNet', 'iNaturalist2017'):
        split = 'train' if kwargs['train'] else 'val'
        del kwargs['train']
        kwargs['split'] = split
    elif dataset_name in ('SVHN', 'ImageNet_LT', 'Places_LT'):
        split = 'train' if kwargs['train'] else 'test'
        del kwargs['train']
        kwargs['split'] = split
    # if dataset_name in ('ImageNet_LT', 'Places_LT'):
    #     kwargs['data_root'] = data_root_map[dataset_name]
    dataset = _dataset_class_map[dataset_name](**kwargs)

    return dataset


def get_dataset_class_number(dataset_name):
    return _dataset_class_number_map[dataset_name]
