#################################
import argparse
from models.model_USCHAD import *
from models.Generators_USCHAD import *
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from utils.experimental_utils import *
import torch.nn.functional as F
from utils.read_data import id2data, id2data_2d_array
from utils.LOSS import kdloss, DiversityLoss, mmd
from utils.augmentation_np import RandAugment
from torchmetrics.classification import MulticlassF1Score, MulticlassCohenKappa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main(args):

    #  select the clients
    train_subject_path = args.train_subject_path
    subject_split_list = load_yaml_to_dict(args.subject_split_file_path)
    train_subject_list = subject_split_list['train']

    augmentations = RandAugment(2)

    ############################################################################
    # define the models
    ############################################################################
    slmodels = list()
    base_1 = ResNet(input_channel=1, num_classes=args.classes).to(device)
    base_2 = CNN(input_channel=1, num_classes=args.classes).to(device)
    base_3 = CNN_tiny(input_channel=1, num_classes=args.classes).to(device)

    [slmodels.append(copy.deepcopy(base_1).to(device)) for _ in range(args.clients_h)]
    [slmodels.append(copy.deepcopy(base_2).to(device)) for _ in range(6)]
    [slmodels.append(copy.deepcopy(base_3).to(device)) for _ in range(args.clients-args.clients_h-6)]
    trainable_params1 = sum(p.numel() for p in base_1.parameters(recurse=True) if p.requires_grad) / 1000000
    trainable_params2 = sum(p.numel() for p in base_2.parameters(recurse=True) if p.requires_grad) / 1000000
    trainable_params3 = sum(p.numel() for p in base_3.parameters(recurse=True) if p.requires_grad) / 1000000
    print("MODEL PARAMS: {}, {}, {}".format(trainable_params1, trainable_params2, trainable_params3))

    generator_base = Generator2(z_size=args.z_latent_dim, input_feat=args.window_size*args.d, fc_layers=args.fc_layers, fc_units=args.fc_units).to(device)
    generators = [copy.deepcopy(generator_base).to(device) for _ in range(args.clients)]
    generator_global = copy.deepcopy(generator_base).to(device)
    discriminator_base = Discriminator2(hidden_dim=args.window_size*args.d, output_dim=1).to(device)
    discriminators = [copy.deepcopy(discriminator_base).to(device) for _ in range(args.clients)]

    ############################################################################
    # define the losses
    ############################################################################
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    diversity_loss = DiversityLoss('l2')
    f1_score = MulticlassF1Score(num_classes=args.classes).to(device)
    cohen_kappa = MulticlassCohenKappa(num_classes=args.classes).to(device)

    ############################################################################
    # define the optimizers
    ############################################################################
    sl_optimizers = list()
    gen_optimizers = list()
    dis_optimizers = list()

    for i in range(args.clients):
        local_sl_optimzier = optim.Adam(slmodels[i].parameters(), lr=0.001)
        sl_optimizers.append(local_sl_optimzier)
        local_gen_optimizer = optim.Adam(generators[i].parameters(), lr=0.0005, betas=(0.5, 0.9999))
        gen_optimizers.append(local_gen_optimizer)
        local_dis_optimizer = optim.Adam(discriminators[i].parameters(), lr=0.0005, betas=(0.5, 0.9999))
        dis_optimizers.append(local_dis_optimizer)

    Correlation = torch.ones((args.clients, args.clients))
    train_label_list = list()
    train_unlabel_list = list()
    for i in range(args.clients):
        id = train_subject_list[i]
        train_data_list_name = 'sample_id_train.txt'
        train_data_list_path = os.path.join(train_subject_path, id, train_data_list_name)
        with open(train_data_list_path, 'r') as f:
            train_label_list_l = f.readlines()
        f.close()
        train_label_list.append(train_label_list_l)

        train_data_unlabel_list_name = 'sample_id_train_unlabel.txt'
        train_data_unlabel_list_path = os.path.join(train_subject_path, id, train_data_unlabel_list_name)
        with open(train_data_unlabel_list_path, 'r') as f:
            train_unlabel_list_l = f.readlines()
        f.close()
        train_unlabel_list.append(train_unlabel_list_l)

    STEPS = float('inf')
    for client in range(args.clients):
        num_unlabel = len(train_unlabel_list[client])
        n_steps_unlabel = int(np.ceil(num_unlabel / args.bs))
        if n_steps_unlabel < STEPS:
            STEPS = n_steps_unlabel

    def training(current_round, Correlation):

        for client in range(args.clients):
            slmodels[client].train()
            generators[client].train()
            discriminators[client].train()

        #######################################
        # -- Phase 0: supervised training
        #######################################

        overall_mean, overall_std = [], []
        Mean_acc, Mean_f1, Mean_kappa = 0.0, 0.0, 0.0
        for _ in range(1):

            ACC, F1, KAPPA = [], [], []
            for client in range(args.clients):

                id = train_subject_list[client]
                abs_path = os.path.join(args.train_subject_path, id)

                # #####################################
                # -- supervised learning
                # #####################################
                train_data, train_label = id2data_2d_array(abs_path, train_label_list[client], [args.window_size, args.d],
                                                           args.all_label, overall_mean, overall_std)

                train_data_unlabel, _ = id2data_2d_array(abs_path, train_unlabel_list[client], [args.window_size, args.d],
                                                         args.all_label, overall_mean, overall_std)
                train_data_augment = augmentations(train_data_unlabel)

                num = train_data.shape[0]
                num_unlabel = train_data_unlabel.shape[0]

                train_data = torch.from_numpy(train_data).to(device, dtype=torch.float32).reshape(num, 1, args.window_size, args.d)
                train_label = torch.from_numpy(train_label).to(device, dtype=torch.int64)

                train_data_unlabel = torch.from_numpy(train_data_unlabel).to(device, dtype=torch.float32).reshape(
                    num_unlabel, 1, args.window_size, args.d)
                train_data_augment = torch.from_numpy(train_data_augment).to(device, dtype=torch.float32).reshape(
                    num_unlabel, 1, args.window_size, args.d)

                n_steps = int(np.ceil(num/args.bs))
                n_steps_unlabel = STEPS
                n_steps_unlabel = n_steps_unlabel - 1 if n_steps_unlabel > 1 else n_steps_unlabel

                # print('# ################################## {}-th Model'.format(client))
                for local_epoch in range(args.l_epochs):
                    shuffle_index = torch.randperm(num)
                    train_data = train_data[shuffle_index, :, :, :]
                    train_label = train_label[shuffle_index, :]

                    shuffle_index_unlabel = torch.randperm(num_unlabel)
                    train_data_unlabel = train_data_unlabel[shuffle_index_unlabel, :, :, :]
                    train_data_augment = train_data_augment[shuffle_index_unlabel, :, :, :]
                    for n in range(n_steps_unlabel):

                        sl_optimizers[client].zero_grad()

                        mod_1 = n % n_steps
                        start_1 = mod_1 * args.bs
                        end_1 = min((mod_1 + 1) * args.bs, num)
                        batch_data = train_data[start_1:end_1, :, :, :]
                        batch_data = batch_data.to(device, dtype=torch.float32)
                        batch_label = train_label[start_1:end_1, :].reshape(-1, ).to(device)

                        _, prediction = slmodels[client](batch_data)
                        cross_e_loss = ce_loss(prediction, batch_label)

                        mod = n % n_steps_unlabel
                        start = mod * args.bs
                        end = min((mod + 1) * args.bs, num_unlabel)

                        batch_data_unlabel = train_data_unlabel[start:end, :, :, :]
                        batch_data_unlabel = batch_data_unlabel.to(device, dtype=torch.float32)
                        batch_data_augment = train_data_augment[start:end, :, :, :]
                        batch_data_augment = batch_data_augment.to(device, dtype=torch.float32)
                        batch_cat = torch.cat((batch_data_unlabel, batch_data_augment), dim=0)
                        num_unsup = batch_data_unlabel.size()[0]
                        embeddings, output = slmodels[client](batch_cat)
                        embeddings_raw = embeddings[:num_unsup, :]
                        embeddings_aug = embeddings[num_unsup:, :]
                        loss_unsup = mse_loss(embeddings_raw, embeddings_aug)

                        loss = cross_e_loss + loss_unsup
                        loss.backward()
                        sl_optimizers[client].step()

                    if (local_epoch + 1) % args.l_epochs == 0 and (current_round % args.print_f == 0 or current_round == args.rounds - 1):
                        acc, f1, kappa = evaluate_l_model(slmodels[client], id, overall_mean, overall_std,
                                                          args.window_size, args.d, args.all_label, args.classes,
                                                          client, train_subject_path)
                        ACC.append(acc)
                        F1.append(f1)
                        KAPPA.append(kappa)

                # train the discriminator and generator
                if current_round < args.rounds:
                    if client < args.clients_h:
                        train_data_unlabel, _ = id2data_2d_array(abs_path, train_unlabel_list[client], [args.window_size, args.d],
                                                                 args.all_label, overall_mean, overall_std)
                        num = train_data_unlabel.shape[0]
                        train_data_unlabel = torch.from_numpy(train_data_unlabel).to(device, dtype=torch.float32).reshape(num, -1)
                        n_steps_unlabel = int(np.ceil(num / args.bs))

                        g_f = args.g_f
                        iterations = int(np.ceil(10 * args.l_epochs * n_steps_unlabel / g_f)) - 1
                        iterations_gen = iterations * g_f
                        count, count_dis, D_LOSS, G_LOSS = 0, 0, 0, 0
                        for local_epoch in range(10*args.l_epochs):

                            shuffle_index = torch.randperm(num)
                            train_data_unlabel = train_data_unlabel[shuffle_index, :]
                            for n in range(n_steps_unlabel):
                                mod = n % n_steps_unlabel
                                start = mod * args.z_batch_size
                                end = min((mod + 1) * args.z_batch_size, num)
                                batch_data = train_data_unlabel[start:end, :]
                                batch_data = batch_data.to(device, dtype=torch.float32)

                                ones = torch.ones((batch_data.size()[0], 1)).to(device)
                                zeros = torch.zeros((batch_data.size()[0], 1)).to(device)

                                z = torch.randn(batch_data.size()[0], 100).to(device)
                                gen_data = generators[client](z)

                                flattened_batch_data = batch_data.reshape(batch_data.size()[0], -1).to(device)
                                d_real = discriminators[client](flattened_batch_data)
                                d_fake = discriminators[client](gen_data.detach())
                                d_loss_1 = mse_loss(d_real, ones)
                                d_loss_2 = mse_loss(d_fake, zeros)
                                d_loss = (d_loss_1 + d_loss_2) * 0.5

                                if count_dis < iterations and count % g_f == 0:
                                    dis_optimizers[client].zero_grad()
                                    d_loss.backward()
                                    dis_optimizers[client].step()
                                    count_dis += 1

                                d_fake_1 = discriminators[client](gen_data)
                                reshaped_gen_data = gen_data.reshape(gen_data.size()[0], 1, args.window_size, args.d)
                                _, s_prob = slmodels[client](reshaped_gen_data)
                                prob = F.softmax(s_prob, dim=1).mean(dim=0)
                                loss_information_entropy = (prob * torch.log10(prob)).sum()
                                task_loss = mse_loss(d_fake_1, ones)
                                gen_loss = task_loss + args.alpha * loss_information_entropy

                                if count < iterations_gen:
                                    gen_optimizers[client].zero_grad()
                                    gen_loss.backward()
                                    gen_optimizers[client].step()
                                    count += 1

            if current_round % args.print_f == 0 or current_round == args.rounds - 1:
                acc_tensor = torch.tensor(ACC)
                f1_tensor = torch.tensor(F1)
                kappa_tensor = torch.tensor(KAPPA)
                mean_acc = torch.mean(acc_tensor)
                mean_f1 = torch.mean(f1_tensor)
                mean_kappa = torch.mean(kappa_tensor)
                Mean_acc = mean_acc
                Mean_f1 = mean_f1
                Mean_kappa = mean_kappa
                print('Current round: {}, Average accuracy: {}, F1 score: {}, KAPPA: {}'.format(
                    current_round+1, mean_acc, mean_f1, mean_kappa))

        GEN_data = torch.zeros((0, args.window_size*args.d)).to(device)
        GEN_labels = list()
        [GEN_labels.append(torch.zeros((0, args.classes)).to(device)) for _ in range(args.clients)]
        similarity = torch.zeros((args.clients, args.clients)).to(device)
        for client in range(args.clients_h):
            for _ in range(4):
                z = torch.randn(args.z_batch_size, 100).to(device)
                generator_global.load_state_dict(generators[client].state_dict())
                dp(generator_global)
                gen_data = generator_global(z)
                GEN_data = torch.cat((GEN_data, gen_data.detach()), dim=0)
                current_labels = list()
                [current_labels.append(torch.zeros((args.z_batch_size, args.classes)).to(device)) for _ in range(args.clients)]
                with torch.no_grad():
                    for client_ in range(args.clients):
                        reshaped_gen_data = gen_data.reshape(gen_data.size()[0], 1, args.window_size, args.d)
                        _, local_pred = slmodels[client_](reshaped_gen_data)
                        GEN_labels[client_] = torch.cat((GEN_labels[client_], local_pred.detach()), dim=0)
                        current_labels[client_] = local_pred
                for client_1 in range(args.clients):
                    for client_2 in range(args.clients):
                        a = current_labels[client_1]
                        b = current_labels[client_2]
                        sim = kdloss(a, b)
                        similarity[client_1:(client_1+1), client_2:(client_2+1)] += sim

        mean = torch.mean(similarity, dim=1, keepdim=True)
        indices = torch.nonzero(similarity < mean)
        correlation = torch.zeros((args.clients, args.clients))
        correlation[indices[:, 0], indices[:, 1]] = 1

        if current_round > 0:
            if args.EMA:
                Correlation = (1-args.multiplier)*Correlation + args.multiplier*correlation
            else:
                Correlation = correlation

        num = GEN_data.size()[0]
        n_steps = int(np.ceil(num / args.z_batch_size))

        if current_round < (args.rounds-1):
            for client in range(args.clients):
                id = train_subject_list[client]
                epochs = int(np.ceil(0.25*args.l_epochs))
                for local_epoch in range(epochs):
                    shuffle_index = torch.randperm(num)
                    GEN_data = GEN_data[shuffle_index, :]
                    for client_1 in range(args.clients):
                        GEN_labels[client_1] = GEN_labels[client_1][shuffle_index, :]

                    for n in range(n_steps):
                        mod = n % n_steps
                        start = mod * args.bs
                        end = min((mod + 1) * args.bs, num)
                        batch_data = GEN_data[start:end, :]
                        batch_data = batch_data.reshape(batch_data.size()[0], 1, args.window_size, args.d)

                        batch_labels = list()
                        [batch_labels.append(GEN_labels[client_1][start:end, :].to(device)) for client_1 in range(args.clients)]

                        sl_optimizers[client].zero_grad()
                        _, prediction = slmodels[client](batch_data)

                        s_loss = torch.zeros((1,)).to(device)
                        for s_clients in range(args.clients):
                            sum = torch.sum(Correlation[client, :]).item()
                            weight = Correlation[client, s_clients].item() / sum
                            s_loss += weight * kdloss(prediction, batch_labels[s_clients])
                        loss = s_loss
                        loss.backward()
                        sl_optimizers[client].step()

        if (current_round + 1) % 5 == 0:
            for client in range(args.clients):
                for param_group in sl_optimizers[client].param_groups:
                    param_group['lr'] = 0.5 * param_group['lr']

        return Correlation, Mean_acc, Mean_f1, Mean_kappa

    ACC, F1, KAPPA = [], [], []
    for current_round in range(args.rounds):
        "obtain initial accuracy"
        if current_round == 0:
            accs, f1s, kappas = [], [], []
            for client in range(args.clients):
                id = train_subject_list[client]
                acc, f1, kappa = evaluate_l_model(slmodels[client], id, [], [],
                                                  args.window_size, args.d, args.all_label, args.classes,
                                                  client, train_subject_path)
                accs.append(acc)
                f1s.append(f1)
                kappas.append(kappa)
            acc_tensor = torch.tensor(accs)
            f1_tensor = torch.tensor(f1s)
            kappa_tensor = torch.tensor(kappas)
            mean_acc = torch.mean(acc_tensor)
            mean_f1 = torch.mean(f1_tensor)
            mean_kappa = torch.mean(kappa_tensor)
            print('CURRENT ROUND: {}, ACC: {}, F1:{}, KAPPA:{}'.format(current_round, mean_acc, mean_f1, mean_kappa))

        Correlation_new, acc, f1, kappa = training(current_round, Correlation)
        ACC.append(acc)
        F1.append(f1)
        KAPPA.append(kappa)
        Correlation = Correlation_new


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ###########################################
    # Experimental setting
    ###########################################
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--subject_split_file_path', type=str, default='../../datasplit/USCHAD/train-test-split')
    parser.add_argument('--train_subject_path', type=str, default='../../datasplit/USCHAD/10/train-subject')
    parser.add_argument('--clients', type=int, default=13)
    parser.add_argument('--clients_h', type=int, default=3)
    parser.add_argument('--d', type=int, default=6)  # features we choose
    parser.add_argument('--window_size', type=int, default=512)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--l_epochs', type=int, default=20)
    parser.add_argument('--classes', type=int, default=12)
    parser.add_argument('--all_label', default=[i for i in range(12)])
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--EMA',type=str, default=True)
    parser.add_argument('--multiplier', type=float, default=0.7)
    parser.add_argument('--z_latent_dim', type=int, default=100)
    parser.add_argument('--z_batch_size', type=int, default=128)
    parser.add_argument('--fc_units', type=int, default=512)
    parser.add_argument('--fc_layers', type=int, default=4)
    parser.add_argument('--g_f', type=int, default=25)
    parser.add_argument('--print_f', type=int, default=5)

    args = parser.parse_args()

    main(args)
