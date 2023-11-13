import torch.optim as optim
import torch.optim.lr_scheduler
import os
import logging
import numpy as np
from utils.global_vars import dtype
from src.dataset import HDF5_Dataset_transpose_uint8, HDF5_Dataset_transpose
import src.models as models
from src.loss import Charbonnier_loss, Recall_Guided_RegLoss, Precision_Guided_RegLoss
from utils.util import make_dir, read_file_list, weights_init_He_normal, Config, find_test_image_h5, evaluate_detection_network
import sys
import time
import argparse


# Set the default parameters
pretrained_model = None
current_directory = os.getcwd()
config = Config()
config.name = 'PLWE'
config.beta1 = 0.9
config.beta2 = 0.999
config.learning_rate = 0.001
config.fake_weight = 1
config.weight_lr = 1
config.bias_lr = 0.1
config.iterations = 1000000
config.stepsize = 400000
config.gamma = 0.1
config.weight_decay = 0.00001
config.batch_size = 64
config.test_batchsize = 64
config.n_train_log = 100
config.n_test_log = 1000
config.timestamp = config.name + '_' + time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
config.test_folder = current_directory + '/test_result/' + config.timestamp + '/'
config.n_save_model = 10000
config.save_folder = current_directory + '/models/' + config.timestamp
config.save_folder += '/' + config.name
config.log_folder = current_directory + '/logs/'
config.train_fake_pos_file = 'train_pl_pos.txt'
config.train_fake_neg_file = 'train_pl_neg.txt'
config.test_file = 'test.txt'


# Set the command line argument parser
parser = argparse.ArgumentParser(description="Whistle extraction experiments")
parser.add_argument('--lamda', type=float)
parser.add_argument('--lamda_recall', type=float)
parser.add_argument('--lamda_prec', type=float)
parser.add_argument('--thres_recall', type=float, default=None)
parser.add_argument('--thres_prec', type=float, default=None)
parser.add_argument('--gamma_recall', type=int, default=1)
parser.add_argument('--gamma_prec', type=int, default=1)
parser.add_argument('--recall_grad', action='store_true')
parser.add_argument('--prec_grad', action='store_true')
for key, value in config.__dict__.items():
    parser.add_argument('--%s' % (key), type=type(value), default=value)
args = parser.parse_args()
for key, value in config.__dict__.items():
    setattr(config, key, getattr(args, key))
make_dir(config.test_folder)
make_dir(config.save_folder)
make_dir(os.path.dirname(config.log_folder))

# read the h5 file list from the arguments
train_filelist_fake_pos = read_file_list(args.train_fake_pos_file)
train_filelist_fake_neg = read_file_list(args.train_fake_neg_file)
test_filelist = read_file_list(args.test_file)


# Configure logger
logging.getLogger('PIL').setLevel(logging.CRITICAL)
logging.basicConfig(filename=os.path.join(config.log_folder, config.timestamp + '.log'), filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
logging.info("Start program")
logging.info(train_filelist_fake_neg)
logging.info(train_filelist_fake_pos)
for attr, value in args.__dict__.items():
    logging.info('args.%s: %s' % (attr, str(value)))

# Register dataset
train_samples = 0
test_samples = 0
train_dataset_fake_neg = HDF5_Dataset_transpose_uint8(hdf5_list=train_filelist_fake_neg, batchsize=config.batch_size // 2)
train_dataset_fake_pos = HDF5_Dataset_transpose_uint8(hdf5_list=train_filelist_fake_pos, batchsize=config.batch_size // 2)
test_dataset = HDF5_Dataset_transpose(hdf5_list=test_filelist, batchsize=config.batch_size)
train_samples += len(train_dataset_fake_pos) + len(train_dataset_fake_neg)
test_samples += len(test_dataset)
test_batch_num, test_pic_num = find_test_image_h5(test_dataset, config)


# Register network
net = models.Detection_ResNet_BN(width=32).type(dtype)
net.apply(weights_init_He_normal)
all_net = net
net_dict = net.state_dict()
if not pretrained_model is None:
    logging.info('Loading %s' % (pretrained_model))
    pretrained_dict = torch.load(pretrained_model)
    for key in net_dict:
        if key in pretrained_dict:
            net_dict[key] = pretrained_dict[key]
    all_net.load_state_dict(net_dict)
logging.info(net)


# Register optimizer and scheduler
config.epoch_size = train_samples / config.batch_size
optimizer = optim.Adam(all_net.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=1e-8, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.stepsize), gamma=config.gamma)

logging.info("Learning rate: " + str(config.learning_rate))
logging.info("Step size: " + str(config.stepsize))
logging.info("Batch size: " + str(config.batch_size))

# Test before training
iterations = 0
epoch = 0
all_net.eval()
psnr_list = evaluate_detection_network(net, test_dataset, config, iterations, test_batch_num, test_pic_num)
mean_psnr = np.mean(psnr_list)
logging.info('Validation###[epoch ' + str(epoch) + " iter " + str(iterations) + ']: mean psnr : ' + str(mean_psnr))
all_net.train()

# Start Training.
logging.info("Start Training ...")
loss_list = []
loss_list_pl = []
loss_list_recall = []
loss_list_prec = []
recall_list = []
precision_list = []
s_time = time.time()
loss_func = Charbonnier_loss()
pos_reg_loss = Recall_Guided_RegLoss(threshold=args.thres_recall, gamma=args.gamma_recall)
neg_reg_loss = Precision_Guided_RegLoss(threshold=args.thres_prec, gamma=args.gamma_prec)

while True:
    all_net.train()
    current_lr = optimizer.param_groups[0]['lr']
    mse_total_loss = torch.zeros(1).type(dtype)
    iterations += 1

    sample_batched_fake_neg = next(train_dataset_fake_neg)
    sample_batched_fake_pos = next(train_dataset_fake_pos)
    sample_batched_fake = [np.concatenate((sample_batched_fake_pos[0], sample_batched_fake_neg[0]), axis=0),
                           np.concatenate((sample_batched_fake_pos[1], sample_batched_fake_neg[1]), axis=0)]
    data_size = sample_batched_fake[0].shape[0]
    shuffle_idx = np.random.permutation(data_size)
    sample_batched_fake[0] = sample_batched_fake[0][shuffle_idx, ...]
    sample_batched_fake[1] = sample_batched_fake[1][shuffle_idx, ...]
    fake_size = sample_batched_fake[0].shape[-1]
    input = torch.from_numpy(np.asarray(sample_batched_fake[0])).type(dtype)
    label = torch.from_numpy(np.asarray(sample_batched_fake[1])).type(dtype)
    output = net(input)
    loss_pl = args.lamda * loss_func(output, label)
    loss_recall, recall = pos_reg_loss(output, label, recall_grad=args.recall_grad)
    loss_recall = args.lamda_recall * loss_recall
    loss_prec, precision = neg_reg_loss(output, label, recall_grad=args.prec_grad)
    loss_prec = args.lamda_prec * loss_prec
    loss = loss_pl + loss_recall + loss_prec

    loss.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()
    all_net.zero_grad()

    loss_list.append(loss.cpu().detach().numpy())
    loss_list_pl.append(loss_pl.cpu().detach().numpy())
    loss_list_recall.append(loss_recall.cpu().detach().numpy())
    loss_list_prec.append(loss_prec.cpu().detach().numpy())
    recall_list.append(recall.cpu().detach().numpy())
    precision_list.append(precision.cpu().detach().numpy())

    if iterations % config.n_train_log == 0:
        e_time = time.time()
        mean_loss = np.mean(loss_list)
        mean_loss_pl = np.mean(loss_list_pl)
        mean_loss_pos = np.mean(loss_list_recall)
        mean_loss_neg = np.mean(loss_list_prec)
        mean_recall = np.mean(recall_list)
        mean_precision = np.mean(precision_list)
        logging.info(
            "[epoch " + str(epoch) + " iter " + str(iterations) + "]:" + ' lr: ' + str(current_lr) +
            ' loss: ' + str(mean_loss) + ' loss plain: ' + str(mean_loss_pl) + ' loss recall_reg: ' + str(mean_loss_pos)
            + ' loss prec_reg: ' + str(mean_loss_neg) + ' recall: ' + str(mean_recall) + ' prec: ' + str(mean_precision)
            + '  time: ' + str(e_time - s_time))
        loss_list = []
        loss_list_pl = []
        loss_list_recall = []
        loss_list_prec = []
        recall_list = []
        precision_list = []
        s_time = time.time()


    if iterations % config.n_save_model == 0:
        logging.info("Saving model ...")
        torch.save(all_net.state_dict(), config.save_folder+'-iter_'+str(iterations))


    if iterations % config.n_test_log == 0:
        logging.info("Testing ...")
        all_net.eval()
        psnr_list = evaluate_detection_network(net, test_dataset, config, iterations, test_batch_num,
                                                     test_pic_num)
        mean_psnr = np.mean(psnr_list)
        logging.info(
            'Validation###[epoch ' + str(epoch) + " iter " + str(iterations) + ']: mean psnr : ' + str(mean_psnr))
        all_net.train()

    if iterations % config.epoch_size == 0:
        epoch += 1

    if iterations > config.iterations:
        break

logging.info('Train epoch' + ' : ' + str(epoch))
