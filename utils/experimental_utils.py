import yaml
import os
import torch
import torch.distributions as tdist
from utils.read_data import id2data, id2data_2d_array
from torchmetrics.classification import MulticlassF1Score, MulticlassCohenKappa


def load_yaml_to_dict(path):

    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def organize_list(subject_path, known_label, unknown_label, num_labeled):

    label_list = []
    unlabel_list = []

    num_known_class = len(known_label)
    num_unknown_class = len(unknown_label)
    a = []
    for i in range(num_known_class):
        a.append(0)

    for files in os.listdir(subject_path):

        filename = files.split('.')[0]

        if 'subject' in filename.split('10'):

            str_label = filename.split('_')[2]
            label = int(str_label)

            if int(label) in known_label or int(label) in unknown_label:
                if label in known_label:
                    t = known_label.index(label)
                    if a[t] >= num_labeled:
                        unlabel_list.append(files)
                    else:
                        a[t] = a[t] + 1
                        label_list.append(files)
                else:
                    unlabel_list.append(files)

    return label_list, unlabel_list

def dp(model, mean=0.0, std=1.0):
    for key in model.state_dict().keys():
        if model.state_dict()[key].dtype == torch.int64:
            continue
        else:
            std_value = torch.std(model.state_dict()[key])
            if std_value.item() == 'nan':
                temp = model.state_dict()[key]
                model.state_dict()[key].data.copy_(temp)
            else:
                nn = tdist.Normal(torch.tensor([mean]), std * torch.std(model.state_dict()[key].detach().cpu()))
                noise = nn.sample(model.state_dict()[key].size()).squeeze()
                noise = noise.to('cuda')
                temp = model.state_dict()[key] + noise
                model.state_dict()[key].data.copy_(temp)

def evaluate_l_model(l_model, id, overall_mean, overall_std, window_size, d, all_label, classes,
                     client, train_subject_path,  device='cuda', LSTM=False, clients_others=0):

    f1_score = MulticlassF1Score(num_classes=classes).to(device)
    cohen_kappa = MulticlassCohenKappa(num_classes=classes).to(device)

    l_model.eval()
    with torch.no_grad():
        test_data_list_name = 'sample_id_test.txt'
        test_data_list_path = os.path.join(train_subject_path, id, test_data_list_name)
        with open(test_data_list_path, 'r') as f:
            test_list = f.readlines()
        f.close()
        abs_path = os.path.join(train_subject_path, id)
        test_data, test_label = id2data(abs_path, test_list, [window_size, d], all_label, overall_mean, overall_std)
        test_data = test_data.reshape(len(test_list), 1, window_size, d)
        if LSTM:
            if client >= clients_others:
                test_data = test_data.reshape(len(test_list), window_size, d)
        test_data = test_data.to(device)

        test_label = test_label.to(device).reshape(-1, )
        _, output = l_model(test_data)
        pred = output.data.max(1)[1]
        correct = pred.eq(test_label).sum().item()
        acc = correct / len(test_list)
        f1 = f1_score(output, test_label)
        kappa = cohen_kappa(output, test_label)
    l_model.train()

    return acc, f1, kappa

