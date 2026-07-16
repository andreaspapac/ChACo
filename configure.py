import argparse
# from Tools.Datasets import *
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# from Models import *
import datetime
from hierarchical import *
from Tools.Datasets import *
from prp_imgnet200 import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_config(args, model, model_id, stem='WAN'):
    # Get current date and time
    now = datetime.datetime.now()
    date_time = now.strftime("%d%m%y_%H%M")
    log_file = './TrRes/x_' + model_id + str(date_time) + '_log.txt'


    # Prepare the log entry
    log_entry = f"Experiment conducted on: {date_time}\n"
    log_entry += "Configuration:\n"
    log_entry += f"  stem_arch: {stem}\n"
    log_entry += f"  channels_list: {model.out_channels_}\n"
    log_entry += f"  conf_flow: {model.flow}\n"
    for arg, value in vars(args).items():
        log_entry += f"  {arg}: {value}\n"

    log_entry += f"  start_end: {model.start_end}\n"
    log_entry += f"  dropout_rates: {model.dropout_rates}\n"

    log_entry += "-" * 50 + "\n"

    if isinstance(model, Hier_CwC_WAN) or isinstance(model, Hier_CwC_ResNet):
        if isinstance(model, Hier_CwC_ResNet):
            log_entry += f"  skip_to: {model.skip_to}\n"
            log_entry += f"  skip_from: {model.skip_from}\n"
            log_entry += f"  skip_mode: {model.skip_mode}\n"
        log_entry += f"  num_supergroups_layers: {model.num_supergroups_layers}\n"
        log_entry += f"  downsample: {model.downsample}\n"
        log_entry += f"  beta_start: {model.beta_start}\n"
        log_entry += f"  beta_end: {model.beta_end}\n"


    try:
        log_entry += f"  maxpool: {model.maxpool}\n"
    except:
        print('No Maxpool.')

    # Write to log file
    with open(log_file, "a") as f:
        f.write(log_entry)


def parse_args():
    parser = argparse.ArgumentParser(description="Configure the hyperparameters")
    # parser.add_argument("--conf_flow", default='RCB', type=str,
    #                     help="Configuration Flow: {'RCB', 'CBR', 'CRB'}")
    parser.add_argument("--data_path", default='C:/Users/Andreas/Desktop/PhD/ChACo/src/sdata/', type=str,
                        help="Data Path")
    parser.add_argument("--seeds", default=[22], type=list, help="Torch  Random Seed")   # 52/13/22/2
    parser.add_argument("--loss_criterion", default='CwC_CE', type=str,
                        help="Loss function: {'CwC', 'CwC_Ortho', 'CwC_CE', 'PvN', 'CWG'}")
    parser.add_argument("--stem", default='WAN', type=str,
                        help="stem architectures: {'WAN', 'ResNet', 'ResNet17'}")
    parser.add_argument("--dev_num", default=0, type=int,
                        help="GPU device number: 0, 1, .. etc.")
    parser.add_argument("--dataset", default='CIFAR10', type=str,
                        help="Dataset: {MNIST, FMNIST, CIFAR10, CIFAR100, STL10}")
    parser.add_argument("--ILT", default='Acc', type=str,
                        help="ILT Strategy: {Acc, Fast}")
    parser.add_argument("--save", default='True', type=str,
                        help="Save Weights")
    parser.add_argument("--CFSE", default='False', type=str,
                        help="CFSE Architecture: {True=CFSE, False=FFCNN}")
    parser.add_argument("--sf_pred", default='False', type=str,
                        help="Enable Sf Predictor: {True=Avg+Sf, False=Sf}")
    parser.add_argument("--ClassGroup", default='False', type=str,
                        help="Class Grouping - Used for CIFAR100")
    parser.add_argument("--retrain", default='False', type=str,
                        help="Retrain Model: {True=Load Weights, False=from Scratch}")
    parser.add_argument("--stage", default=1, type=int,
                        help="Training Stage: {0 = from scratch}")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch Size")
    # parser.add_argument("--num_workers", default=4, type=int)
    #                              self.maxpool = [False, False, True, False, True, False, False, True, False]
    parser.add_argument("--skip_mode", default="cat", type=str,
                        help="Skip type: {cat, add, add_pad, none}")
    # parser.add_argument("--channels_list", default=[100, 200, 400, 400, 800, 1600], type=list,
    #                     help="Number of Channels per Layer")  # [40, 80, 160, 320] #[20, 80, 240, 480] [60, 120, 240, 480, 960, 1920] [80, 320, 1280]
    parser.add_argument("--show_iters", default=400, type=int,
                        help="Iteration frequency for Visualizing Layerwise Stats")
    parser.add_argument("--n_epochs", default=230, type=int,
                        help="Total Number of Training Epochs")
    parser.add_argument("--N_Classes", default=10, type=int,
                        help="Number of Classes")
    parser.add_argument("--min_testerror", default=0.55, type=float,
                        help="Minimum Testing Error for Weight Saving")
    parser.add_argument("--load_epoch", default=12, type=int,
                        help="Model Weights Epoch")
    parser.add_argument("--log_root", default="./runs", type=str, help="Root folder for runs")
    parser.add_argument("--log_steps", default=50, type=int, help="Log step metrics every N steps")
    parser.add_argument("--online_logger", default="tensorboard", type=str,
                        help="Online logger backend: {'tensorboard', 'none'}")
    parser.add_argument("--run_name", default="", type=str,
                        help="Optional human-readable run name prefix")
    parser.add_argument("--downsample_mode", default="auto", type=str,
                        help="Shortcut/downsample mode: {'auto', 'avgpool', 'maxpool', 'stride'}")

    args = parser.parse_args()

    args.CFSE = str2bool(args.CFSE)
    args.save = str2bool(args.save)
    args.retrain = str2bool(args.retrain)
    args.sf_pred = str2bool(args.sf_pred)
    args.ClassGroup = str2bool(args.ClassGroup)

    return args


def configure(args):

    if args.CFSE:
        architecture = 'CFSE'
    else:
        architecture = 'FFCNN'
    if args.sf_pred:
        preds = 'AvgSf'
    else:
        preds = 'Avg'

    if args.dataset[:5] == 'STL10':
        print('STL-10')
        train_data = STL10(args.data_path, split='train', download=True)
        test_data = STL10(args.data_path, split='test', download=True)
        train_dataset = X_STL(train_data, number_samples=5000)
        test_dataset = X_STL(test_data, number_samples=5000)

    elif args.dataset[:5] == 'CIFAR':
        print('==> Preparing data..')
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)],
        #         p=0.8
        #     ),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                         (0.2023, 0.1994, 0.2010)),
        # ])
        #
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                         (0.2023, 0.1994, 0.2010)),
        # ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply(
            #             [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)],
            #             p=0.8
            #         ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == 'CIFAR10':
            print('CIFAR-10')
            train_dataset = CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
            test_dataset = CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
        else:
            print('CIFAR10-100')
            train_dataset = CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
            test_dataset  = CIFAR100(args.data_path, train=False, download=True, transform=transform_test)

            # args.ClassGroup = True
            # args.channels_list = [60, 120, 240, 400, 800, 1600]
            args.N_Classes = 100
            # args.sf_min_testerror = 0.6
            # args.n_epochs = 200

    elif args.dataset in ["TINYIMAGENET", "TINYIMAGENET200", "tiny-imagenet-200", "imgnet200"]:
        print("Tiny-ImageNet-200")
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        root = os.path.join("./../../HebbGate/src/data/", "tiny-imagenet-200")
        # root = os.path.join(args.data_path, "tiny-imagenet-200")
        train_dir = os.path.join(root, "train")
        val_fixed_dir = os.path.join(root, "val")
        # val_fixed_dir = prepare_tinyimagenet_val(root)

        train_dataset = ImageFolder(train_dir, transform=transform_train)
        test_dataset = ImageFolder(val_fixed_dir, transform=transform_test)  # use val as test

        args.N_Classes = 200

    elif args.dataset in ['MNIST', 'FMNIST']:
        if args.dataset == 'MNIST':
            train_data = MNIST(args.data_path, train=True, download=True)
            test_data = MNIST(args.data_path, train=False, download=True)
        elif args.dataset == 'FMNIST':
            train_data = FashionMNIST(args.data_path, train=True, download=True)
            test_data = FashionMNIST(args.data_path, train=False, download=True)
        train_dataset = X_MNIST(train_data, number_samples=60000, dataset=args.dataset)
        test_dataset = X_MNIST(test_data, number_samples=10000, dataset=args.dataset)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")



    return args, architecture, preds, train_dataset, test_dataset


    # #  # Manual Overwrite below
    # parser.add_argument("--save", action='store_true', default=False,
    #                     help="Save Weights")
    # parser.add_argument('--CFSE', action='store_true', default=False,
    #                     help='Set this flag to True to enable CFSE')
    # parser.add_argument("--sf_pred", action='store_true', default=False,
    #                     help="Enable Sf Predictor: {True=Avg+Sf, False=Sf}")
    # parser.add_argument("--ClassGroup", action='store_true', default=False,
    #                     help="Class Grouping - Used for CIFAR100")
    # parser.add_argument("--retrain", action='store_true', default=False,
    #                     help="Retrain Model: {True=Load Weights, False=from Scratch}")


def configure_test(args):

    # # Evaluation - Manual # #


    if args.CFSE:
        architecture = 'CFSE'
    else:
        architecture = 'FFCNN'
    if args.sf_pred:
        preds = 'AvgSf'
    else:
        preds = 'Avg'

    if args.dataset[:5] == 'CIFAR10':
        if args.dataset == 'CIFAR10':
            print('CIFAR10-10')
            # train_data = CIFAR10(args.data_path, train=True, download=True)
            test_data = CIFAR10(args.data_path, train=False, download=True)
        else:
            print('CIFAR10-100')
            # train_data = CIFAR100(args.data_path, train=True, download=True)
            test_data = CIFAR100(args.data_path, train=False, download=True)
            args.ClassGroup = True
            args.channels_list = [60, 120, 240, 400, 800, 1600]
            args.N_Classes = [20, 20, 20, 20, 100, 100]
            args.sf_min_testerror = 0.6

        # train_dataset = X_CIFAR(train_data, number_samples=50000)
        test_dataset = X_CIFAR(test_data, number_samples=10000)

    else:
        if args.dataset == 'MNIST':
            # train_data = MNIST(args.data_path, train=True, download=True)
            test_data = MNIST(args.data_path, train=False, download=True)
        elif args.dataset == 'FMNIST':
            # train_data = FashionMNIST(args.data_path, train=True, download=True)
            test_data = FashionMNIST(args.data_path, train=False, download=True)
        # train_dataset = X_MNIST(train_data, number_samples=60000, dataset=args.dataset)
        test_dataset = X_MNIST(test_data, number_samples=10000, dataset=args.dataset)

    return args, architecture, preds, test_dataset



    # #  # Manual Overwrite below
    # parser.add_argument("--save", action='store_true', default=False,
    #                     help="Save Weights")
    # parser.add_argument('--CFSE', action='store_true', default=False,
    #                     help='Set this flag to True to enable CFSE')
    # parser.add_argument("--sf_pred", action='store_true', default=False,
    #                     help="Enable Sf Predictor: {True=Avg+Sf, False=Sf}")
    # parser.add_argument("--ClassGroup", action='store_true', default=False,
    #                     help="Class Grouping - Used for CIFAR100")
    # parser.add_argument("--retrain", action='store_true', default=False,
    #                     help="Retrain Model: {True=Load Weights, False=from Scratch}")


