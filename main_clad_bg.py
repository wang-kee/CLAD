import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
import wandb

from src.model import Neg_Sample_Dictionary, resnet_9l
from src.dataloader import (parallel_dataloader,
                            load_datasets, 
                            load_testsets)
from src.utils import (set_seed, 
                       shuffle_fg_index, 
                       cal_contrastive_loss, 
                       construct_pos_samples)
from load_eval import eval_model

def args_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--N_neg_samples', default=32, type=int,
                        help='Number of negative samples')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--nb_epochs', default=60, type=int,
                        help='total number of training epochs')    
    parser.add_argument('--lambd', default=1, type=float,
                        help='weight for auxiliary contrastive loss')
    parser.add_argument('--tau', default=0.2, type=float,
                        help='temperature parameter in contrasitve loss')
    parser.add_argument('--with_con_loss', default=1, type=int,
                        help='flag to include contrastive loss (set to 1 for CLAD)')
    parser.add_argument('--with_pos_loss', default=0, type=int,
                        help='flag to include supervised loss for positive samples (set to 1 for CLAD+)')
    parser.add_argument('--debug', default=0, type=int,
                        help='debugging flag')
    parser.add_argument('--imagenet_pretrained_model_dir', default='model_weights/imagenet_pretrained_resnet50_weights.pkl', type=str,
                        help='direction to load the imagenet-pretrained model backbone')    
    parser.add_argument('--train_from_scratch', default=0, type=int,
                        help='flag to train the model from scratch instead of loading imagenet-pretrained weights')    
        
    return parser.parse_args()



def main():
    args = args_parse()
    set_seed(42)
    
    # load the respective datasets from disk
    data_train, data_train_fg, data_train_bg, data_val, data_val_randbg, data_val_samebg = load_datasets()
    
    # create dataloader to load the datasets in parallel 
    # e.g. data_train and data_train_fg always have the same foreground, data_train and data_train_bg always have the same background
    train_dataset = parallel_dataloader(data_train, data_train_fg, data_train_bg)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)
    
    val_dataset = parallel_dataloader(data_val, data_val_randbg, data_val_samebg)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=SequentialSampler(val_dataset),
                            num_workers=4)
    
    # load imagenet-pretrained model
    imagenet_pretrained_model = resnet_9l()
    if not args.train_from_scratch:
        imagenet_pretrained_model.load_state_dict(torch.load(args.imagenet_pretrained_model_dir))
    
    # imagenet_pretrained_model = torch.load(args.imagenet_pretrained_model)
    
    device = torch.device('cuda')
    model, criterion = imagenet_pretrained_model.to(device), nn.CrossEntropyLoss().to(device)
    
    n_batch = int(len(data_train) / args.batch_size)
    
    # use Adam optimizer
    optimizer = torch.optim.Adam([{'params':model.parameters(), 'lr':1e-3}])
    # lr decays after 20 epochs of training
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
    
    val_acc_opt = 0 # initialize best validation accuracy
    
    # get model name for current experiment
    if args.with_con_loss and not args.with_pos_loss:
        model_name = "CLAD"
    elif args.with_con_loss and args.with_pos_loss:
        model_name = "CLAD+"
    elif not args.with_con_loss and not args.with_pos_loss:
        model_name = "Baseline_model"
    else:
        model_name = "Customized_model"
    
    # model save direction
    model_save_dir_best = '../../../scratch/izar/kewang/bg_models/{}_n{}_lambd{}_best.pkl'\
        .format(model_name, args.N_neg_samples, args.lambd)
    model_save_dir_last = '../../../scratch/izar/kewang/bg_models/{}_n{}_lambd{}_last.pkl'\
        .format(model_name, args.N_neg_samples, args.lambd)
        
    print(f"Training starts for {model_name}, "
          f"lambda = {args.lambd}, "
          f"number of negative samples: {args.N_neg_samples}, "
          f"temperature parameter: {args.tau}")
          
    for e in range(args.nb_epochs):

        model.train()
        train_loss, train_acc, train_acc_pos = 0, 0, 0
        train_loss_sup, train_loss_con = 0, 0
        
        # the anchor is loaded in parrell with its pre-segmented foreground and background
        for b, ((inputs, target), (fg, _), (bg, _)) in enumerate(train_loader):
            
            if args.debug and b == 2:
                break
            
            optimizer.zero_grad()
            target_copy = target.detach().clone()
            
            inputs, target = inputs.to(device), target.to(device)
            # forward pass for anchor
            latent, logits = model(inputs)
            # calculate classification loss for the anchor
            classification_loss = criterion(logits, target)
            
            # generate positive samples, by 1) creating a shuffled copy of the batch 
            # and 2) mixing the anchors' foreground and the shuffled copy's background
            shuffle_index = shuffle_fg_index(target_copy)
            pos_samples = construct_pos_samples(fg, bg[shuffle_index], augmentation=True).to(device)
            
            # forward pass for positive samples
            latent_pos, logits_pos = model(pos_samples)
            
            # for CLAD+: include supervised loss for positive samples as well
            if args.with_pos_loss: 
                classification_loss += criterion(logits_pos, target)
            
            # for CLAD or CLAD+: include auxiliary contrastive loss
            if args.with_con_loss:
                # initialize negative samples dictionary at the beginning of the training
                if e==0 and b==0:
                    neg_sample_dic = Neg_Sample_Dictionary(target_copy, latent, n_neg_samples=args.N_neg_samples)
                
                # update negative samples dictionary with the latent of generated positive samples
                neg_sample_dic.update_dict(target_copy[shuffle_index], latent_pos)
                # calculate contrastive loss
                contrastive_loss = cal_contrastive_loss(latent, latent_pos, target_copy, neg_sample_dic.dic, args)
                
                loss = classification_loss + args.lambd * contrastive_loss
            else:
                loss = classification_loss
            
            train_loss += loss.item()
            train_loss_sup += classification_loss.item()
            if not args.with_con_loss:
                train_loss_con = 0
            else:
                train_loss_con += contrastive_loss.item()
            
            train_acc += (logits.max(dim=1)[1] == target).float().mean().item()
            train_acc_pos += (logits_pos.max(dim=1)[1] == target).float().mean().item()
            
            loss.backward()
            optimizer.step()
            print("\rEpoch: {:d} batch: {:d} / {} loss: {:.4f} | {:.2%}".format(e+1, b, n_batch, loss, b*1.0/n_batch), end='', flush=True)
        scheduler.step()
        
        # validate model
        model.eval()
        val_loss, val_loss_randbg = 0, 0
        val_acc, val_acc_randbg, val_acc_samebg = 0, 0, 0
        
        for b, ((inputs, target), (inputs_randbg, _), (inputs_samebg, _)) in enumerate(val_loader):
            
            if args.debug and b == 2:
                break
            
            inputs, inputs_randbg, inputs_samebg, target = \
                inputs.to(device), inputs_randbg.to(device), inputs_samebg.to(device), target.to(device)

            with torch.no_grad():
                _, logits = model(inputs)
                _, logits_pos = model(inputs_randbg)
                _, logits_samebg = model(inputs_samebg)

                val_loss += criterion(logits, target).item()
                val_loss_randbg += criterion(logits_pos, target).item()
                val_acc += (logits.max(dim=1)[1] == target).float().mean().item()
                val_acc_randbg += (logits_pos.max(dim=1)[1] == target).float().mean().item()
                val_acc_samebg += (logits_samebg.max(dim=1)[1] == target).float().mean().item()

        train_loss, train_acc, train_acc_pos = \
            train_loss/len(train_loader), train_acc/len(train_loader), train_acc_pos/len(train_loader)
        val_loss, val_loss_randbg, val_acc, val_acc_randbg, val_acc_samebg = \
            val_loss/len(val_loader), val_loss_randbg/(len(val_loader)), val_acc/len(val_loader), val_acc_randbg/len(val_loader), val_acc_samebg/len(val_loader)

        wandb.log(
            {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_acc_pos': train_acc_pos,
                'val_loss': val_loss,
                'val_loss_randbg': val_loss_randbg,
                'val_acc': val_acc,
                'val_acc_randbg': val_acc_randbg,
                'val_acc_samebg': val_acc_samebg
            }
        )
        
        # save the best model on validation accuracy of random-background images
        if val_acc_randbg >= val_acc_opt:
            val_acc_opt = val_acc_randbg
            torch.save(model.state_dict(), model_save_dir_best)
        torch.save(model.state_dict(), model_save_dir_last)
        
        print('\n----------------------- Epoch {} -----------------------'.format(e + 1))
        print(f'Model: {model_name} '
              f'Train_loss: {train_loss:.4f}, '
              f'supervised loss: {train_loss_sup/len(train_loader):.4f}, '
              f'contrastive loss: {train_loss_con/len(train_loader):.4f} | '
              f'Train_acc (anchor / positive samples): {train_acc:.4f}, {train_acc_pos:.4f} | '
              f'Val_loss (original / random-bg images): {val_loss:.4f}, {val_loss_randbg:.4f} | '
              f'Val_acc (original / random-bg / same-bg images): {val_acc:.4f}, {val_acc_randbg:.4f}, {val_acc_samebg:.4f} | '
              f'BG-gap: {val_acc_samebg - val_acc_randbg:.4f}')
    
    # evaluate model after training complete
    test_loader, test_loader_fg, test_loader_randbg, test_loader_samebg, test_loader_bg = load_testsets()
    eval_model(model, test_loader, test_loader_fg, test_loader_randbg, test_loader_samebg, test_loader_bg)
    
if __name__ == '__main__':
    wandb.init(project="bg_influence_proj", entity="wangke", name="clad")
    main()