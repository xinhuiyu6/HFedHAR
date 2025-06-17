import numpy as np
import torch
from torch.nn.functional import interpolate
import random
from scipy.interpolate import interp1d

class Normalize:
    """
    Normalizes based on mean and std. Used by skeleton and inertial modalities
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

class ToTensor:
    def __call__(self, x):
        return torch.tensor(x)

class Permute:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return x.permute(self.shape)

class ToFloat:
    def __call__(self, x):
        return x.float()

class Normalize_0_1:
    def __call__(self, x):
        # min_x = x.min(dim=2).values
        # max_x = x.max(dim=2).values
        min_x = torch.min(x, dim=2, keepdim=True).values
        max_x = torch.max(x, dim=2, keepdim=True).values
        t = (x - min_x) / (max_x - min_x)
        return t

class Resampling():
    def __call__(self, x):
        x_reduced_dim = x.resize(x.size()[0], x.size()[1], x.size()[2])
        x_permute = torch.permute(x_reduced_dim, dims=[0, 2, 1])  # (batch_size, time_steps, features)
        x_arr = np.array(x_permute)

        M, N = random.choice([[2, 1], [3, 2]])
        time_steps = x_arr.shape[1]
        raw_set = np.arange(x_arr.shape[1])
        interp_steps = np.arange(0, raw_set[-1] + 1e-1, 1 / (M + 1))
        x_interp = interp1d(raw_set, x_arr, axis=1)
        x_up = x_interp(interp_steps)

        length_inserted = x_up.shape[1]
        start = random.randint(0, length_inserted - time_steps * (N + 1))
        index_selected = np.arange(start, start + time_steps * (N + 1), N + 1)
        x_aug = x_up[:, index_selected, :]
        x_tensor = torch.tensor(x_aug).resize(x.size()[0], x.size()[2], x.size()[1], 1)
        x_tensor = torch.permute(x_tensor, dims=[0, 2, 1, 3])
        return x_tensor


class Normalize_Gaussian():
    def __call__(self, x):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        normalized_data = (x - mean) / std
        return normalized_data


class Jittering():
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def __call__(self, x):
        noise = np.random.normal(loc=0, scale=self.sigma, size=x.shape)
        x = x + noise
        return x

class Scaling():
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x):
        x = np.expand_dims(x, axis=0)
        factor = np.random.normal(loc=1., scale=self.sigma, size=(x.shape[0], x.shape[2]))
        output = np.multiply(x, factor[:, np.newaxis, :])
        return np.squeeze(output, axis=0)


class Rotation():
    def __init__(self):
        pass

    def __call__(self, x):
        x = np.expand_dims(x, axis=0)
        flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
        rotation_axis = np.arange(x.shape[2])
        np.random.shuffle(rotation_axis)
        output = flip[:, np.newaxis, :] * x[:, :, rotation_axis]
        return np.squeeze(output, axis=0)


class ChannelShuffle():
    def __init__(self):
        pass

    def __call__(self, x):
        rotate_axis = np.arange(x.shape[1])
        np.random.shuffle(rotate_axis)
        return x[:, rotate_axis]


class Permutation():
    def __init__(self, max_segments=5):
        self.max_segments = max_segments

    def __call__(self, x):
        orig_steps = np.arange(x.shape[0])
        num_segs = np.random.randint(1, self.max_segments)

        ret = np.zeros_like(x)
        if num_segs > 1:
            splits = np.array_split(orig_steps, num_segs)
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret = x[warp]
        else:
            ret = x
        return ret