import argparse

from fed_train_contrl import main

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
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=20, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=2, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=5, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=20, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=3, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=20, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', default=False, help="run in train+validation mode (default: False)")
#parser.add_argument('--data.sequential', action='store_true', default=False, help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', default=False, help="run in CUDA mode (default: False)")

# model args
default_model_name = 'midifew_preteacher_conv2d'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default='1,11,11', metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=128, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")


# train args
parser.add_argument('--train.epochs', type=int, default=1, metavar='NEPOCHS',
                    help='number of epochs to train local model (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=2, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
default_weight_decay = 0.0001
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.patience', type=int, default=200, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')
parser.add_argument('--train.isDistill', action='store_true', default=False,
                    help='Knowledge Distill(default: False)')

# sec model setting
parser.add_argument('--train.sec_weight_decay', type=float, default=0.00, metavar='SEC_WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.sec_learning_rate', type=float, default=0.008, metavar='SEC_LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--model.midiFew', action='store_true', default=True,
                    help='use midiFew(default: False)')

# log args
default_fields = 'loss,acc,Precision,Recall,F1'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = '../fed_train/results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))


# fed args
parser.add_argument('--fed.client_num', type=int, default=5, metavar='NCLIENTS',
                    help='number of clients to train (default: 5)')
parser.add_argument('--fed.comm_num', type=int, default=50, metavar='NCOMMUNICATION',
                    help='number of communication (default: 100)')
parser.add_argument('--fed.cfraction', type=float, default=0.6, metavar='CFRACTION',
                    help='number of fraction in clients to train (default: 0.6)')
parser.add_argument('--fed.save_freq', type=int, default=2,
                    help='global model save frequency(of communication)')
parser.add_argument('--fed.val_freq', type=int, default=1,
                    help='validate global model frequency(of communication)')
default_global_model_path = '../../scripts/results'
parser.add_argument('--fed.global_save_path', type=str, default=default_global_model_path, metavar='GLOBALPATH',
                    help="location of global model to save (default: {:s})".format(default_global_model_path))
parser.add_argument('--fed.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")

#file_suffix
suffix_name = 'globalteacherFT'
parser.add_argument('--file.suffixName', type=str, default=suffix_name, metavar='SUFFIXNAME',
                    help="fileSuffix name (default: {:s})".format(suffix_name))

args = vars(parser.parse_args())
print(args.values())
main(args)
