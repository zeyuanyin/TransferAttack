import torch
import torch.nn.functional as F

from ...gradient.mifgsm import MIFGSM
from ...utils import *

from .sko.GA import GA

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
        1) The code only support batchsize = 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        """
        simulated models
        1 (ResNet50)
        2 (ResNet50 and VGG16)
        3 (ResNet50, VGG16, and DenseNet161)
        """
        self.simulated_models = [ models.__dict__[model_name](weights="DEFAULT") for model_name in ['resnet50', 'vgg16', 'densenet161'] ]

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

    def predict_transfer_score(self, x, img, label):
        # 每个个体的得分，通过每个mask单独作用到图像进行对抗攻击产生对抗样本在一组黑盒模型上的效果得分获得
        mask = torch.from_numpy(x)
        mask = mask.reshape(-1,int(img_height/patch_size),int(img_width/patch_size))
        # X_adv = single_attack(img,mask,label)  # TODO
        numsum = x.shape[0]
        scorelist = []
        bn = int(np.ceil(numsum/batch_size))
        # print(bn)
        # print(batch_size)
        # print(mask.shape)
        # print(img.shape)
        for i in range(bn):
            bs = i*batch_size
            be = min((i+1)*batch_size, numsum)
            bn = be-bs
            X_adv = batch_attack(torch.stack([img]*bn), mask[bs:be], torch.stack([label]*bn), white_models)
            scorelist = np.append(scorelist,score_transferability(X_adv, torch.stack([label]*bn),gray_models))
        return scorelist



    def learn_mask(self, data, label):
        # Format the predict/callback functions for the differential evolution algorithm
        def myfunc(x,img=data,label=label):
            # 评估种群得分的函数
            return self.predict_transfer_score(x,data,label)


        lb = [0] * len(bounds)
        ub = [elem[1] for elem in bounds]
        # 用的GA模板改成了DE算法
        de = MyDE(func=myfunc, n_dim=len(bounds), size_pop=popsize, max_iter=maxiter, prob_mut=0.001, lb=lb, ub=ub, precision=1, img=None, label=None)


        masks, y = de.run()

    # def learn_mask(self, data, label):
    #     """
    #     Differential Evolution to learn the mask

    #     Arguments:
    #         data (1, C, H, W): tensor for input images
    #         labels (N,): tensor for ground-truth labels
    #     """
    #     T_DE = 5 # number of iterations for differential evolution
    #     P = 40 # population size
    #     Rho = 0.3 # crossover probability
    #     num_masked_patch = 12 # number of masked patches
    #     # number of all patches = 10*10 = 100

    #     M = [ self.random_mask(num_masked_patch) for _ in range(P) ] # initialize the population
    #     feedback_current = [ self.get_feedback(data, mask, label) for mask in M ]

    #     for k in range(T_DE):
    #         ### Crossover Strategy
    #         # get lowest Rho*P feedback values' indices
    #         indices = torch.argsort(torch.tensor(feedback_current))[:int(Rho*P)]

    #         mask_next = []
    #         for _ in range(int(Rho*P)):
    #             # randomly select 2 masks from the lowest Rho*P feedback values' indices
    #             idx1, idx2 = torch.randperm(int(Rho*P))[:2]
    #             mask1 = M[indices[idx1]]
    #             mask2 = M[indices[idx2]]
    #             mask_cross = mask1 + mask2
    #             for i in range(10):
    #                 for j in range(10):
    #                     if mask_cross[i, j] == 2:
    #                         mask_cross[i, j] = 1
    #                     elif mask_cross[i, j] == 0:
    #                         mask_cross[i, j] = 0
    #                     else: # mask_cross[i, j] == 1
    #                         mask_cross[i, j] = torch.randint(0, 2, (1,)).item()
    #             mask_next.append(mask_cross)

    #         ### Mutation Strategy
    #         for _ in range(P - int(Rho*P)):
    #             mask_next.append(self.random_mask(num_masked_patch))

    #         ### Selection Strategy
    #         feedback_next = [ self.get_feedback(data, mask, label) for mask in mask_next ]

    #         # get the lowest P feedback values' indices from 2*P feedback values(feedback_current & feedback_next)
    #         feedback_2P = feedback_current + feedback_next
    #         mask_2P = M + mask_next

    #         indices = torch.argsort(torch.tensor(feedback_2P))[:P]

    #         # update the population and feedback values
    #         M = [ mask_2P[idx] for idx in indices ]
    #         feedback_current = [ feedback_2P[idx] for idx in indices ]

    #     # return the best mask w/ lowest feedback value (sorted in ascending order)
    #     return M[0]

    # def random_mask(self, num_patch):
    #     """
    #     Generate a mask matrix of size (10, 10) with `num_patch` zeros

    #     Arguments:
    #         num_patch (int): the number of zeros in the mask matrix
    #     """

    #     mask = torch.ones(10, 10)  # Initialize all elements as 1
    #     indices = torch.randperm(100)[:num_patch]  # Randomly select num_zeros indices
    #     mask.view(-1)[indices] = 0  # Set the selected indices to 0

    #     # print(mask)
    #     return mask

    # def apply_mask(self, data, mask):
    #     """
    #     Apply patch-wise mask to data

    #     Arguments:
    #         data (1, C, H, W): tensor for input images
    #         mask (10, 10): tensor for mask, 0/1
    #     """
    #     _, _, height, width = data.shape
    #     patch_height = height // 10
    #     patch_width = width // 10

    #     # split to 10x10 patches
    #     for i in range(10):
    #         for j in range(10):
    #             start_h = i * patch_height
    #             end_h = (i + 1) * patch_height
    #             start_w = j * patch_width
    #             end_w = (j + 1) * patch_width

    #             if mask[i, j] == 0: # if the block is masked
    #                 data[:, :, start_h:end_h, start_w:end_w] = 0
    #     return data

    # def get_feedback(self, data, mask, label):
    #     masked_data = self.apply_mask(data, mask)
    #     # for simplicity, we use the IFGSM object's forward function to get the adversarial perturbation
    #     perturbation = super().forward(masked_data, label)
    #     adv_data = data + perturbation
    #     ce_criterion = torch.nn.CrossEntropyLoss()
    #     ce_loss = []
    #     for model in self.simulated_models:
    #         # get the logits on simulated model
    #         logits = model(adv_data)
    #         loss = ce_criterion(logits, label)
    #         ce_loss.append(loss.item())

    #     loss_mean = np.mean(ce_loss)
    #     loss_var = np.var(ce_loss)

    #     return - loss_mean + loss_var



    # modified from official code: https://github.com/zhaoshiji123/LPM


class MyDE(GA):
    # 可自定义排序，杂交，变异，选择
    def ranking(self):
        self.Chrom = self.Chrom[np.argsort(self.Y),:]
        self.Y = self.Y[(np.argsort(self.Y))]

    def crossover(self):
        Chrom, size_pop, len_chrom = self.Chrom, self.size_pop, self.len_chrom
        generation_best_index = self.Y.argmin()
        best_chrom = self.Chrom[generation_best_index]
        best_chrom_Y = self.Y[generation_best_index]
        scale_inbreeding = 0.3
        cross_chrom_size = int(scale_inbreeding * self.size_pop)
        # print(cross_chrom_size)
        superior_size = int(0.3 * self.size_pop)
        generation_superior = self.Chrom[:superior_size,:]
        # half_size_pop = int(size_pop / 2)
        # Chrom1, Chrom2 = self.Chrom[:size_pop,:][:half_size_pop], self.Chrom[:size_pop,:][half_size_pop:]
        self.crossover_Chrom = np.zeros(shape=(cross_chrom_size, len_chrom), dtype=int)
        # print(self.crossover_Chrom.shape)
        for i in range(cross_chrom_size):
            n1 = np.random.randint(0, superior_size, 2)
            # print(n1.shape)
            while n1[0] == n1[1]:
                n1 = np.random.randint(0, superior_size, 2)
            # 让 0 跟多一些
            check_1 = 1
            check_2 = 0
            for j in range(self.len_chrom):
                if generation_superior[n1[0]][j] == 1 and generation_superior[n1[1]][j] == 1:
                    self.crossover_Chrom[i][j] = 1
                elif generation_superior[n1[0]][j] == 0 and generation_superior[n1[1]][j] == 0:
                    self.crossover_Chrom[i][j] = 0
                elif generation_superior[n1[0]][j] == 1 and generation_superior[n1[1]][j] == 0:
                    self.crossover_Chrom[i][j] = generation_superior[n1[check_1]][j]
                    check_1 = 1 - check_1
                elif generation_superior[n1[0]][j] == 0 and generation_superior[n1[1]][j] == 1:
                    self.crossover_Chrom[i][j] = generation_superior[n1[check_2]][j]
                    check_2 = 1 - check_2
        return self.crossover_Chrom


    def mutation(self):
        scale_inbreeding = 0.3 #+ self.iter/self.max_iter*(0.8-0.2)
        rate = 0.1
        middle_1 = np.zeros((int(self.size_pop*(1-scale_inbreeding)), int(rate * self.len_chrom)))
        middle_2 = np.ones((int(self.size_pop*(1-scale_inbreeding)),self.len_chrom - int(rate * self.len_chrom)))
        self.mutation_Chrom = np.concatenate((middle_1,middle_2), axis=1)
        for i in range(self.mutation_Chrom.shape[0]):
            self.mutation_Chrom[i] = np.random.permutation(self.mutation_Chrom[i])
        return self.mutation_Chrom


    def selection(self, tourn_size=3):
        '''
        greedy selection
        '''
        # 上一代个体Chrom,得分self.Y
        # 得到这一代个体以及分数
        offspring_Chrom = np.vstack((self.crossover_Chrom,self.mutation_Chrom))
        f_offspring  = self.func(offspring_Chrom)
        # f_chrom = self.Y.copy()
        print("this generate score:")
        print(f_offspring)
        num_inbreeding = int(0.3 * self.size_pop)
        selection_chrom = np.vstack((offspring_Chrom, self.Chrom))
        selection_chrom_Y = np.hstack((f_offspring, self.Y))
        # print(selection_chrom_Y)
        generation_best_index = selection_chrom_Y.argmin()
        # print(selection_chrom[generation_best_index])


        a, indices = np.unique(selection_chrom_Y, return_index=True)
        # print(a)
        # print(indices)

        selection_chrom_1 = np.zeros_like(selection_chrom[0:len(a)])
        selection_chrom_1 = selection_chrom[indices]
        # selection_chrom = selection_chrom[np.argsort(selection_chrom_Y),:]
        # selection_chrom_Y = selection_chrom_Y[(np.argsort(selection_chrom_Y))]
        # print("selection_chrom_1")
        # print(selection_chrom_1)
        if len(a) >= self.size_pop:
            self.Chrom = selection_chrom_1[:self.size_pop,:]
            self.Y = a[:self.size_pop]
        else:
            self.Chrom[0: len(a)] = selection_chrom_1[:len(a),:]
            self.Y[0: len(a)] = a[:len(a)]
            self.Chrom[len(a):self.size_pop] = selection_chrom_1[len(a)-1]
            self.Y[len(a):self.size_pop] = a[len(a)-1]
        # print(self.Chrom[0])
        # assert False

