from torchattacks import *
from adv_eval import Evaluator

import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
torch.cuda.empty_cache()
np.random.seed(42)

# Default GPU configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='imagenet')

    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for model evaluation",
    )
    parser.add_argument(
        "--atk_type",
        choices=[
            "clean",
            "apgd",
            "cc",
        ],
        default="clean",
        help="Evaluation/Attack type",
    )

    args = parser.parse_args()
    param = vars(args)
    

    '''
    Set Dataloaders and path
    '''
    
    if args.dataset == 'imagenet':
        data_path = '/datasets2/ILSVRC2012_val/'
        test_transform = None # Add transforms for test set
        test_dataset = datasets.ImageFolder(data_path + '/val', transform=test_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    else:
        raise NotImplementedError('No other dataset implemented yet')


    """
	Prepping Model
	"""
    print("==> Building model..")

    model_path = None # Add model path here
    save_path = None # Add save path here
    net = None

    net = torch.nn.DataParallel(net)
    net = None # Load model here   
    net.to(device)

    """
	Evaluations and Attacks
	"""

    print("==> Evaluating model..")
    net = net.eval()
    eval = Evaluator(device, net)

    '''
    Running Clean Evaluation
    '''
    if args.atk_type == "clean":
        print("Running Clean Evaluation")
        clean_acc = 100 * eval.clean_accuracy(test_loader)
        print("Clean Accuracy: {}".format(clean_acc))
        print("Completed!! \n\n")

    '''
    Running Common Corruption Evaluation
    '''
    if param["atk_type"] == "cc":
        if args.dataset.lower() in "imagenet":
            print("Running Common Corruption Evaluation on Imagenet-C")
            data_path = "/SCRATCH2/machiraj/BIRD_data/Imagenet-C/"
            cc_accs, corruptions_list = eval.common_corruptions(
                data_path,
                data_transforms=test_transform,
                batch_size=args.batch_size,
            )

        # Average per corruption across 5 severity levels
        avg_data = [
            [key, value]
            for (key, value) in zip(cc_accs.keys(), cc_accs.values())
            if "avg" in key
        ]

        print("Common Corruption Average Accuracy: {}".format(avg_data))

        # Corruption accuracies for all severity level
        for corr in corruptions_list:
            if corr == "natural":
                pass
            else:
                corr_data = [
                    [key, value]
                    for (key, value) in zip(cc_accs.keys(), cc_accs.values())
                    if corr in key
                ]

                # print(corr, corr_data)
                print("Corruption: {} Accuracy: {}".format(corr, corr_data))

        print("Completed!! \n\n")

    '''
    Running Adversarial L-inf PGD Evaluation
    '''
    if args.atk_type == "pgd":
        for eps_num in args.epsilon_list:
            print("Running PGD Evaluation for epsilon = ", eps_num)
            eps = eps_num / 255.0
            # print(eps)
            pgd_acc = 100 * eval.attack_pgd(test_loader, eps, nb_iter=7, eot_iter=1)
            print("PGD Accuracy: {}".format(pgd_acc))
        
    '''
    
    Running Adversarial L-inf APGD Evaluation
    Note:
    *  APGD is a stronger version of PGD, but is slower

    '''

    if args.atk_type == "apgd":
        for eps_num in args.epsilon_list:
            print("Running APGD Evaluation for epsilon = ", eps_num)
            eps = eps_num / 255.0

            apgd_acc = 100 * eval.attack_apgd(
                test_loader,
                eps,
                nb_iter=7,
                eot_iter=1, # more than one if there is any stochasticity/randomness in the model
            )
            print("APGD Accuracy: {}".format(apgd_acc))