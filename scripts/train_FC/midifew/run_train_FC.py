import argparse

from train_FC import main

parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
default_dataset = 'pipeline'
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
default_split = 'vinyals'
parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--data.way', type=int, default=2, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=10, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=20, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=2, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=10, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=20, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=20, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=20, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', default=False, help="run in train+validation mode (default: False)")
#parser.add_argument('--data.sequential', action='store_true', default=False, help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', default=False, help="run in CUDA mode (default: False)")

# model args
default_model_name = 'FC_Net'
#default_model_name = 'midifew_final'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))

default_exp_dir = '../../results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))
#best_model_name
default_best_name = 'best_FC_Net.pt'
parser.add_argument('--log.best_name', type=str, default=default_best_name, metavar='BEST_NAME',
                    help="the model name should be saved (default: {:s})".format(default_best_name))


parser.add_argument('--model.x_dim', type=str, default='1,11,11', metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=128, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")

# train args
parser.add_argument('--train.epochs', type=int, default=1000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='SGD', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.sec_learning_rate', type=float, default=0.008, metavar='SEC_LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=10, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
default_weight_decay = 0
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.sec_weight_decay', type=float, default=0.00, metavar='SEC_WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.patience', type=int, default=200, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')
#Distillation
parser.add_argument('--train.isDistill', action='store_true', default=False,
                    help='Knowledge Distillation(default: False)')

# log args
default_fields = 'loss,acc,Precision,Recall,F1'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))

args = vars(parser.parse_args())
print(args.values())
main(args)
