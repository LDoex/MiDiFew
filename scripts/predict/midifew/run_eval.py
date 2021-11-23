import argparse

from eval import main

parser = argparse.ArgumentParser(description='Evaluate few-shot prototypical networks')

default_model_path = '../../../scripts/train/midifew/results/best_midifew.pt'
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))

default_model_path = '../../../scripts/train/midifew/results/best_teacher_model_sec.pt'
parser.add_argument('--model.sec_model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))
parser.add_argument('--model.midiFew', action='store_true', default=False,
                    help='use midiFew(default: False)')

parser.add_argument('--data.test_way', type=int, default=2, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as model's data.test_way (default: 0)")
parser.add_argument('--data.test_shot', type=int, default=5 , metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as model's data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as model's data.query (default: 0)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 1000)")

args = vars(parser.parse_args())

main(args)
