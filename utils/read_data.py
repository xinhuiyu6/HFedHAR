import torch
import os
import numpy as np


def normalize(data):

    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    # mean = np.mean(data, axis=(0, 1)).reshape(1, -1)
    # std = np.std(data, axis=(0, 1)).reshape(1, -1)
    # mean = np.mean(data, axis=(1, 2), keepdims=True)
    # std = np.std(data, axis=(1, 2), keepdims=True)

    return (data-mean)/std

def normalize_return_stat(data):

    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    # mean = np.mean(data, axis=(0, 1)).reshape(1, -1)
    # std = np.std(data, axis=(0, 1)).reshape(1, -1)
    # mean = np.mean(data, axis=(1, 2), keepdims=True)
    # std = np.std(data, axis=(1, 2), keepdims=True)

    return (data-mean)/std, mean, std


def normalize_tensor(data):
    mean = torch.mean(data, dim=0, keepdim=True)
    std = torch.std(data, dim=0, keepdim=True)
    # mean = torch.mean(data, dim=(0, 1)).reshape(1, -1)
    # std = torch.std(data, dim=(0, 1)).reshape(1, -1)
    # mean = torch.mean(data, dim=(1, 2), keepdim=True)
    # std = torch.std(data, dim=(1, 2), keepdim=True)

    normalized_data = (data - mean) / std
    normalized_data = normalized_data.reshape(-1, data.size()[1]*data.size()[2])
    return normalized_data, mean, std


def normalize_tensor_HARBox(data):
    mean = torch.mean(data, dim=0, keepdim=True)
    std = torch.std(data, dim=0, keepdim=True)

    normalized_data = (data - mean) / std
    # normalized_data = normalized_data.reshape(-1, data.size()[1]*data.size()[2])
    return normalized_data, mean, std


def normalzie_0_1_tensor(data):
    min_x = torch.min(data, dim=1, keepdim=True).values
    max_x = torch.max(data, dim=1, keepdim=True).values
    t = (data - min_x) / (max_x - min_x)
    return t


def cal_moments(abs_path, data_list, shape, all_label):
    num = len(data_list)
    d = np.prod(shape)
    data = torch.zeros((num, d))

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        normalized_data = original_array.reshape(-1)
        data_tensor = torch.from_numpy(normalized_data)
        data[count, :] = data_tensor
        count = count + 1

    _, mean, std = normalize_tensor(data)
    return mean, std


def id2data(abs_path, data_list, shape, all_label, mean, std):

    num = len(data_list)
    d = np.prod(shape)
    data = torch.zeros((num, d))
    label = torch.zeros((num, 1), dtype=torch.int64)

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        normalized_data = original_array.reshape(-1)
        data_tensor = torch.from_numpy(normalized_data)
        data[count, :] = data_tensor
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        label[count, :] = int(mapped_label)
        count = count + 1

    data = data.reshape(-1, shape[0], shape[1])
    data, mean, std = normalize_tensor(data)
    return data, label


def id2data_HARBox(abs_path, data_list, shape, all_label, mean, std):

    num = len(data_list)
    d = np.prod(shape)
    data = torch.zeros((num, d))
    label = torch.zeros((num, 1), dtype=torch.int64)

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path)  # .reshape(*shape)
        normalized_data = original_array  # .reshape(-1)
        data_tensor = torch.from_numpy(normalized_data)
        data[count, :] = data_tensor
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        label[count, :] = int(mapped_label)
        count = count + 1

    # data = data.reshape(-1, shape[0], shape[1])
    # data, mean, std = normalize_tensor_HARBox(data)
    return data, label


def id2data_use_global_stat(abs_path, data_list, shape, all_label, mean, std):

    num = len(data_list)
    d = np.prod(shape)
    data = torch.zeros((num, d))
    label = torch.zeros((num, 1), dtype=torch.int64)

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        normalized_data = original_array.reshape(-1)
        data_tensor = torch.from_numpy(normalized_data)
        data[count, :] = data_tensor
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        label[count, :] = int(mapped_label)
        count = count + 1

    data = data.reshape(-1, shape[0], shape[1])
    mean = torch.from_numpy(mean).to(dtype=torch.float32)
    std = torch.from_numpy(std).to(dtype=torch.float32)
    data = (data - mean) / std
    return data, label


def id2data_use_global_stat_HARBox(abs_path, data_list, shape, all_label, mean, std):

    num = len(data_list)
    d = np.prod(shape)
    data = torch.zeros((num, d))
    label = torch.zeros((num, 1), dtype=torch.int64)

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path)  # .reshape(*shape)
        normalized_data = original_array # .reshape(-1)
        data_tensor = torch.from_numpy(normalized_data)
        data[count, :] = data_tensor
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        label[count, :] = int(mapped_label)
        count = count + 1

    # data = data.reshape(-1, shape[0], shape[1])
    # mean = torch.from_numpy(mean).to(dtype=torch.float32)
    # std = torch.from_numpy(std).to(dtype=torch.float32)
    # data = (data - mean) / std
    return data, label


def id2data_2d_array(abs_path, data_list, shape, all_label, mean, std):

    data = np.zeros((0, shape[0], shape[1]))
    label = np.zeros((0, 1))

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        original_array = np.expand_dims(original_array, axis=0)
        data = np.concatenate((data, original_array), axis=0)
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        mapped_label = np.array([[mapped_label]], dtype=int)
        label = np.concatenate((label, mapped_label), axis=0)
        count = count + 1

    data = normalize(data)
    data = data.reshape(-1, shape[0], shape[1])
    return data, label


def id2data_2d_array_HARBox(abs_path, data_list, shape, all_label, mean, std):

    data = np.zeros((0, shape[0]*shape[1]))
    label = np.zeros((0, 1))

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path) # .reshape(*shape)
        original_array = np.expand_dims(original_array, axis=0)
        data = np.concatenate((data, original_array), axis=0)
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        mapped_label = np.array([[mapped_label]], dtype=int)
        label = np.concatenate((label, mapped_label), axis=0)
        count = count + 1

    # data = normalize(data)
    data = data.reshape(-1, shape[0]*shape[1])
    return data, label


def id2data_all_array(train_subject_path, data_list, shape, all_label, mean, std):

    num = len(data_list)
    d = np.prod(shape)
    data = np.zeros((0, shape[0], shape[1]))
    label = np.zeros((0, 1))

    count = 0
    for files in data_list:
        id = files.split('_')[0]
        file_path = os.path.join(train_subject_path, id, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        original_array = np.expand_dims(original_array, axis=0)
        data = np.concatenate((data, original_array), axis=0)
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        mapped_label = np.array([[mapped_label]], dtype=int)
        label = np.concatenate((label, mapped_label), axis=0)
        count = count + 1

    data, mean, std = normalize_return_stat(data)
    data = data.reshape(-1, shape[0], shape[1])

    return data, label, mean, std


def id2data_all_array_HARBox(train_subject_path, data_list, shape, all_label, mean, std):

    data = np.zeros((0, shape[0]*shape[1]))  # shape=[batch_size, 900]
    label = np.zeros((0, 1))

    count = 0
    for files in data_list:
        id = files.split('_')[0]
        file_path = os.path.join(train_subject_path, id, files.split('\n')[0])
        original_array = np.loadtxt(file_path)
        original_array = np.expand_dims(original_array, axis=0)
        data = np.concatenate((data, original_array), axis=0)
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        mapped_label = np.array([[mapped_label]], dtype=int)
        label = np.concatenate((label, mapped_label), axis=0)
        count = count + 1

    # data, mean, std = normalize_return_stat(data)
    data = data.reshape(-1, shape[0]*shape[1])

    return data, label, mean, std


def id2data_all(train_subject_path, data_list, shape, all_label, mean, std):

    # shape = [window_size, d]
    num = len(data_list)
    d = np.prod(shape)
    data = torch.zeros((num, d))
    label = torch.zeros((num, 1), dtype=torch.int64)

    count = 0
    for files in data_list:
        id = files.split('_')[0]
        file_path = os.path.join(train_subject_path, id, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        normalized_data = original_array.reshape(-1)
        data_tensor = torch.from_numpy(normalized_data)
        data[count, :] = data_tensor
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        mapped_label = all_label.index(raw_label)
        label[count, :] = int(mapped_label)
        count = count + 1

    data = normalize(data)
    # data, mean, std = normalize_tensor(data)
    return data, label


def id2data_binary(abs_path, data_list, shape, all_label, mean, std, target_label):

    # shape = [window_size, d]
    num = len(data_list)
    d = np.prod(shape)
    data = torch.zeros((num, d))
    label = torch.zeros((num, 1), dtype=torch.int64)

    count = 0
    for files in data_list:
        file_path = os.path.join(abs_path, files.split('\n')[0])
        original_array = np.loadtxt(file_path).reshape(*shape)
        normalized_data = original_array.reshape(-1)
        # normalized_data = normalize(original_array).reshape(-1)
        data_tensor = torch.from_numpy(normalized_data)
        data[count, :] = data_tensor
        #################
        # -- label map
        #################
        filename = files.split('.')[0]
        raw_label = int(filename.split('_')[1])
        if raw_label == int(target_label):
            mapped_label = 0
        else:
            mapped_label = 1
        label[count, :] = int(mapped_label)
        count = count + 1

    data, mean, std = normalize_tensor(data)
    return data, label


if __name__ == '__main__':
    data = np.array([[1, 2], [3, 4]])
    print(normalize(data))