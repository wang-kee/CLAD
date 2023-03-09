import argparse

import torch
import torch.nn as nn

from src.model import resnet_9l
from src.utils import set_seed
from src.dataloader import load_testsets

def args_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--load_model_weights_dir', default='../../../scratch/izar/kewang/bg_models/Baseline model_n32_lambd1_best.pkl', type=str, help='model load direction')    
    return parser.parse_args()

def eval_model(model, 
               test_loader, 
               test_loader_fg, 
               test_loader_randbg, 
               test_loader_samebg, 
               test_loader_bg):
    
    """ 
    Evaluate the model on the following test set loaders:
        test_loader: original test set
        test_loader_fg: only-foreground test set
        test_loader_randbg: random-background test set
        test_loader_samebg: random-background (from same class) test set
        test_loader_bg: only-background test set
    """
    
    data_loader_names = ['original data', 'foreground data', 'random-background data', 'same-background data', 'only_bg data']
    
    model.eval()
    n_batch_val = int(4550 / 10)
    device = torch.device('cuda')
    
    print('Evaluting model on test set.')
    for i, data_loader in enumerate([test_loader, test_loader_fg, test_loader_randbg, test_loader_samebg, test_loader_bg]):
        
        predictions = torch.tensor([]).to(device)
        targets = torch.tensor([]).to(device)

        with torch.no_grad():
            for b, (inputs, target) in enumerate(data_loader):
                inputs, target = inputs.to(device), target.to(device)
                inter, out = model(inputs)
                pred = out.max(dim=1)[1].to(device)
                predictions = torch.cat((predictions, pred), 0)
                targets = torch.cat((targets, target), 0)
                print("\rbatch: {:d} / {} ".format(b+1, n_batch_val), end='',  flush=True)

            accuracy = (predictions == targets).float().mean().item()
            print(' --- Test accuracy on {} for resnet: {:.4f}'.format(data_loader_names[i], accuracy))

def main():
    args = args_parse()
    set_seed(42)
    
    # load imagenet-pretrained model
    model = resnet_9l()
    model.load_state_dict(torch.load(args.load_model_weights_dir))
    
    device = torch.device('cuda')
    model = model.to(device)
    
    # evaluate the loaded model on test set
    test_loader, test_loader_fg, test_loader_randbg, test_loader_samebg, test_loader_bg = load_testsets()
    eval_model(model, test_loader, test_loader_fg, test_loader_randbg, test_loader_samebg, test_loader_bg)

if __name__ == '__main__':
    main()