import torch
import torch.nn.functional as F

from ..gradient.mifgsm import MIFGSM
from ..utils import EnsembleModel


class AdaEA(MIFGSM):
    """
    AdaEA (Adaptive Ensemble Attack)
    'An Adaptive Model Ensemble Adversarial Attack for Boosting Adversarial Transferability (ICCV) 2023)'(https://arxiv.org/abs/2308.02897)

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

    Official Arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.0, mlp_gamma=0.25 (we follow mlp_gamma=0.5 in official code)

    Example Script:
        python main.py --attack adaea --input_dir ./path/to/data --output_dir adv_data/adaea/ensemble --model='resnet18,resnet101,resnext50_32x4d,densenet121'
    """

    def __init__(self, epoch=20, epsilon=8 / 255, alpha=2 / 255, decay=0.9, **kwargs):
        kwargs["attack"] = "AdaEA"
        self.ensemble_mode = "ind"

        # AdaEA hyperparameters
        self.adaea_beta = 10
        self.threshold = 0

        super().__init__(**kwargs)
        assert isinstance(self.model, EnsembleModel)
        self.model: EnsembleModel = self.model  # for type hinting
        # self.model = torch.nn.DataParallel(self.model)

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
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            logits = self.get_logits(data + delta)  # [#models, BS, #classes]
            loss_list = [self.get_loss(logit, label) for logit in logits]
            grad_list = [self.get_grad(loss, delta) for loss in loss_list]
            delta_list = [self.update_delta(delta, data, grad, self.alpha) for grad in grad_list]

            # AGM
            weight = self.agm(data, delta_list, label)

            # DRF
            cos_res = self.drf(grad_list, data_size=data.shape)
            cos_res[cos_res >= self.threshold] = 1.0
            cos_res[cos_res < self.threshold] = 0.0

            logit_ens = logits * weight.view(self.model.num_models, 1, 1)
            logit_ens = logit_ens.sum(dim=0)
            loss_ens = self.get_loss(logit_ens, label)
            grad_ens = self.get_grad(loss_ens, delta) * cos_res
            momentum = self.get_momentum(grad_ens, momentum)

            delta = self.update_delta(delta, data, momentum, self.alpha)

            # print('ok')
            # exit()

        return delta.detach()


    def get_grad(self, loss, delta, **kwargs):
        return torch.autograd.grad(loss, delta, retain_graph=True, create_graph=False)[0]

    def agm(self, data, delta_list, label):
        """
        Adaptive gradient modulation
        """
        loss_func = torch.nn.CrossEntropyLoss()
        adv_list = [data + delta for delta in delta_list]

        loss_self = []
        for i in range(self.model.num_models):
            loss_self.append(loss_func(self.model.models[i](adv_list[i]), label))

        # loss_cross
        w = torch.zeros(size=(self.model.num_models,), device=self.device)
        for i in range(self.model.num_models):
            for j in range(self.model.num_models):
                if i != j:
                    w[i] += loss_func(self.model.models[i](adv_list[j]), label) / loss_self[j]

        w = torch.softmax(self.adaea_beta * w, dim=0)
        return w

    def drf(self, grad_list, data_size):
        """
        disparity-reduced filter
        """
        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

        reduce_map = torch.zeros(
            size=(self.model.num_models, self.model.num_models, data_size[0], data_size[-2], data_size[-1]),
            dtype=torch.float,
            device=self.device,
        )  # [#models, #models, BS, W, H]

        reduce_map_result = torch.zeros(
            size=(self.model.num_models, data_size[0], data_size[-2], data_size[-1]),
            dtype=torch.float,
            device=self.device,
        )  # [#models, BS, W, H]

        for i in range(self.model.num_models):
            for j in range(self.model.num_models):
                if i < j:
                    reduce_map[i][j] = sim_func(F.normalize(grad_list[i], dim=1), F.normalize(grad_list[j], dim=1))
            if i < j:
                one_reduce_map = (reduce_map[i, :].sum(dim=0) + reduce_map[:, i].sum(dim=0)) / (self.model.num_models - 1)
                reduce_map_result[i] = one_reduce_map

        return reduce_map_result.mean(dim=0).view(data_size[0], 1, data_size[-2], data_size[-1])
