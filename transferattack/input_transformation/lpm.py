import torch
import torch.nn.functional as F

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class LPM(MIFGSM):
    """
    LPM (Learnable Patch-wise Masks)
    'Boosting Adversarial Transferability with Learnable Patch-wise Masks (IEEE MM 2023)'(https://ieeexplore.ieee.org/abstract/document/10251606)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1

    Example script:
        python main.py --attack lpm --output_dir adv_data/lpm/resnet18 --batchsize 1

    NOTE:
        1) The code only support batchsize = 1. It will take about 6 hours on a single 4090 GPU to run the attack on the whole 1000 images.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        """
        simulated models
        1 (ResNet50)
        2 (ResNet50 and VGG16)
        3 (ResNet50, VGG16, and DenseNet161)
        """
        self.simulated_models = [ models.__dict__[model_name](weights="IMAGENET1K_V1") for model_name in ['resnet50', 'vgg16', 'densenet161'] ]

        # self.source_model = self.model

        for model in self.simulated_models:
            model.eval().cuda()
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # only support batch size 1, i.e., one image at a time
        assert  data.shape[0] == 1
        # Learn the mask
        mask = self.learn_mask(data, label)
        data = self.apply_mask(data, mask)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Apply mask and obtain the output
            logits = self.get_logits(self.transform((data+delta), momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def learn_mask(self, data, label):
        """
        Differential Evolution to learn the mask

        Arguments:
            data (1, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels
        """
        T_DE = 5 # number of iterations for differential evolution
        P = 40 # population size
        Rho = 0.3 # crossover probability
        num_masked_patch = 12 # number of masked patches
        # number of all patches = 10*10 = 100

        M = [ self.random_mask(num_masked_patch) for _ in range(P) ] # initialize the population
        feedback_current = [ self.get_feedback(data, mask, label) for mask in M ]

        for k in range(T_DE):
            ### Crossover Strategy
            # get lowest Rho*P feedback values' indices
            indices = torch.argsort(torch.tensor(feedback_current))[:int(Rho*P)]

            mask_next = []
            for _ in range(int(Rho*P)):
                # randomly select 2 masks from the lowest Rho*P feedback values' indices
                idx1, idx2 = torch.randperm(int(Rho*P))[:2]
                mask1 = M[indices[idx1]]
                mask2 = M[indices[idx2]]
                mask_cross = mask1 + mask2
                for i in range(10):
                    for j in range(10):
                        if mask_cross[i, j] == 2:
                            mask_cross[i, j] = 1
                        elif mask_cross[i, j] == 0:
                            mask_cross[i, j] = 0
                        else: # mask_cross[i, j] == 1
                            mask_cross[i, j] = torch.randint(0, 2, (1,)).item()
                mask_next.append(mask_cross)

            ### Mutation Strategy
            for _ in range(P - int(Rho*P)):
                mask_next.append(self.random_mask(num_masked_patch))

            ### Selection Strategy
            feedback_next = [ self.get_feedback(data, mask, label) for mask in mask_next ]

            # get the lowest P feedback values' indices from 2*P feedback values(feedback_current & feedback_next)
            feedback_2P = feedback_current + feedback_next
            mask_2P = M + mask_next

            indices = torch.argsort(torch.tensor(feedback_2P))[:P]

            # update the population and feedback values
            M = [ mask_2P[idx] for idx in indices ]
            feedback_current = [ feedback_2P[idx] for idx in indices ]

        # return the best mask w/ lowest feedback value (sorted in ascending order)
        return M[0]

    def random_mask(self, num_patch):
        """
        Generate a mask matrix of size (10, 10) with `num_patch` zeros

        Arguments:
            num_patch (int): the number of zeros in the mask matrix
        """

        mask = torch.ones(10, 10)  # Initialize all elements as 1
        indices = torch.randperm(100)[:num_patch]  # Randomly select num_zeros indices
        mask.view(-1)[indices] = 0  # Set the selected indices to 0

        # print(mask)
        return mask

    def apply_mask(self, data, mask):
        """
        Apply patch-wise mask to data

        Arguments:
            data (1, C, H, W): tensor for input images
            mask (10, 10): tensor for mask, 0/1
        """
        _, _, height, width = data.shape
        patch_height = height // 10
        patch_width = width // 10

        # split to 10x10 patches
        for i in range(10):
            for j in range(10):
                start_h = i * patch_height
                end_h = (i + 1) * patch_height
                start_w = j * patch_width
                end_w = (j + 1) * patch_width

                if mask[i, j] == 0: # if the block is masked
                    data[:, :, start_h:end_h, start_w:end_w] = 0
        return data

    def get_feedback(self, data, mask, label):
        masked_data = self.apply_mask(data, mask)
        # for simplicity, we use the IFGSM object's forward function to get the adversarial perturbation
        perturbation = super().forward(masked_data, label)
        adv_data = data + perturbation
        ce_criterion = torch.nn.CrossEntropyLoss()
        ce_loss = []
        for model in self.simulated_models:
            # get the logits on simulated model
            logits = model(adv_data)
            loss = ce_criterion(logits, label)
            ce_loss.append(loss.item())

        loss_mean = np.mean(ce_loss)
        loss_var = np.var(ce_loss)

        return - loss_mean + loss_var