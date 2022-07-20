from torchattacks import *
import cc_data as CC
from tqdm import tqdm
from torchvision.models import *
from torchvision.utils import *
import torch
import numpy as np
import sys

torch.manual_seed(42)
torch.cuda.empty_cache()
np.random.seed(42)

class Evaluator:
    def __init__(self, device, model):
        self.device = device
        self.model = model

    # For clean accuracy
    def clean_accuracy(self, clean_loader):
        """ Evaluate the model on clean dataset. """
        self.model.eval()

        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(clean_loader):
                data, target = data.to(self.device), target.to(self.device)
                print(batch_idx)
                # print(target.shape)
                # exit()
                output = self.model(data)
                pred = (output[0] if (type(output) is tuple) else output).argmax(
                    dim=1, keepdim=True
                )
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(clean_loader.dataset)
        print("Clean Test Acc {:.3f}".format(100.0 * acc))
        return acc

    def common_corruptions(
        self,
        data_path,
        data_transforms,
        batch_size=128,
    ):
        corruptions_txt = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]
        accs = dict()
        with tqdm(total=len(corruptions_txt) * 5, ncols=80) as pbar:
            for _, cname in enumerate(corruptions_txt):
                # load dataset
                avg_sev = 0
                for sev in range(1, 6):
                    dataset_obj = CC.ImageNetC(
                        data_path,
                        cname,
                        severity=sev,
                        input_transform=data_transforms,
                        target_transform=None,
                    )
                    dataset = dataset_obj.dataset()

                    testloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, shuffle=False, num_workers=4
                    )

                    correct = 0
                    acc = 0

                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(testloader):
                            data, target = (
                                data.to(self.device),
                                target.to(self.device),
                            )
                            output = self.model(data)
                            pred = output.argmax(dim=1, keepdim=True)
                            correct += pred.eq(target.view_as(pred)
                                                ).sum().item()

                    acc = correct / len(testloader.dataset)
                    accs[f"{cname+'_'+str(sev)}"] = acc

                    avg_sev += acc

                    pbar.set_postfix_str(f"{cname, sev}: {acc}")
                    pbar.update()

                # Get average acc for each corruption across 5 severity levels 
                avg_sev /= 5
                accs[f"{cname + '_avg'}"] = avg_sev

            avg = np.mean(list(accs.values()))
            accs["avg"] = avg

        return accs, corruptions_txt


    def attack_pgd(self, dataloader, epsilon, nb_iter=7, eot_iter=3):
        self.model.eval()
        atk = EOTPGD(
            self.model, eps=epsilon, alpha=2 / 255, steps=nb_iter, eot_iter=eot_iter
        )  # Linf version of EOT PGD
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                with torch.enable_grad():
                    adv_images = atk(data, target).to(self.device)
                output = self.model(adv_images)
                pred = (output[0] if (type(output) is tuple) else output).argmax(
                    dim=1, keepdim=True
                )
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(dataloader.dataset)
        print("PGD attack Acc {:.3f}".format(100.0 * acc))
        return acc

    def attack_apgd(
        self, dataloader, epsilon, nb_iter=7, eot_iter=3):
        self.model.eval()

        atk = APGD(
            self.model,
            norm="Linf",
            eps=epsilon,
            steps=nb_iter,
            eot_iter=eot_iter,
            n_restarts=1,
            loss="ce",
        )
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):

                data, target = data.to(self.device), target.to(self.device)
                with torch.enable_grad():
                    adv_images = atk(data, target).to(self.device)

                output = self.model(adv_images)
                pred = (output[0] if (type(output) is tuple) else output).argmax(
                    dim=1, keepdim=True
                )
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(dataloader.dataset)
        print("APGD attack Acc {:.3f}".format(100.0 * acc))
        return acc