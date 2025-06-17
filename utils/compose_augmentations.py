from torchvision import transforms
from utils.single_augmentations import ChannelShuffle, Jittering, Permutation, Rotation, Scaling
from utils.experimental_utils import load_yaml_to_dict

def compose_random_augmentations(config_dict):

    inertial_augmentations = {
        'jittering': Jittering,
        'scaling': Scaling,
        'rotation': Rotation,
        'permutation': Permutation,
        'channel_shuffle': ChannelShuffle
    }


    all_augmentations = {
        "inertial": inertial_augmentations,
    }

    transforms_list = []
    augmentations_for_modality = all_augmentations["inertial"]
    for key in config_dict:
        if config_dict[key]['apply']:
            if 'parameters' not in config_dict[key]:
                config_dict[key]['parameters'] = {}
            augmentation = augmentations_for_modality[key](**config_dict[key]['parameters'])
            probability = config_dict[key]['probability']
            transforms_list.append(transforms.RandomApply([augmentation], p=probability))
    return transforms_list


if __name__ == '__main__':
    path = '../configs/augmentation'
    config_dict = load_yaml_to_dict(path)
    compose_transforms = compose_random_augmentations(config_dict)

