import argparse
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import tqdm

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# from attack import attack_zoo
from transferattack.utils import *



class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None ,targeted=False, eval=False):
        self.targeted = targeted
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir,filename)

        if not os.path.exists(filepath):
            # 如果文件不存在，返回None或者其他合适的值，表示这个样本无效
            print('donot find file: {}'.format(filepath))
            return None, None, None
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2,0,1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['target_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
        return f2l

def custom_collate(batch):
    # Filter out samples with None values
    batch = [item for item in batch if item[0] is not None]

    if not batch:
        return None

    return list(zip(*batch))


def get_parser():
    parser = argparse.ArgumentParser(description='MIST')
    parser.add_argument('-e', '--eval', action='store_true')
    # parser.add_argument('--attack', default='mist', type=str, help='the attack algorithm',
                        # choices=list(attack_zoo.keys()))
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('-b', '--batchsize', default=1, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for the benign images')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--num_mist', default=30, type=int, help='the number of add-in images for mist')
    parser.add_argument('--num_admix', default=3, type=int, help='the number of admixed images for admix')
    parser.add_argument('--ensemble', default=None, type=str, help='the ensemble of attacks for mist',choices=[None, 'cnn', 'transformer', 'all'])
    parser.add_argument('--result_path', default='results_eval.txt', type=str)
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--reverse', action='store_true')

    return parser.parse_args()

def main():
    args = get_parser()
    print(args)
    # mkdir if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    if args.reverse:
        print('reverse dataset [TODO]')
        # reverse_sampler = ReverseSampler(dataset)
    else:
        reverse_sampler = None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4, sampler=reverse_sampler,collate_fn=custom_collate)


    assert args.batchsize == 1

    not_correct_list_name = []
    not_correct_list_label = []

    if args.eval:
        asr = dict()
        res = '|'
        cnn_model_paper = ['resnet18', 'resnet101', 'resnext50_32x4d', 'densenet121']
        vit_model_paper = ['vit_base_patch16_224.sam_in1k', 'pit_b_224', 'visformer_small', 'swin_tiny_patch4_window7_224']

        # args.targeted = False
        for model_name, model in load_pretrained_model(cnn_model_paper,vit_model_paper):
            model = wrap_model(model.eval().cuda())
            for p in model.parameters():
                p.requires_grad = False
            correct, total = 0, 0

            for batch in dataloader:
                if batch is None:
                    print("Empty batch, skipping...")
                    continue

                images, labels, filenames = batch
                images = torch.stack(images).cuda()
                pred = model(images.cuda())
                is_correct = (np.array(labels) == pred.argmax(dim=1).detach().cpu().numpy()).sum()

                if is_correct == 0:
                    not_correct_list_name.append(filenames)
                    not_correct_list_label.append(labels)

                correct += is_correct

                # correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += images.shape[0]
            if args.targeted: # correct: pred == target_label
                asr[model_name] = (correct / total) * 100
            else: # correct: pred == original_label
                asr[model_name] = (1 - correct / total) * 100
            print(model_name, asr[model_name])
            res += ' {:.1f} |'.format(asr[model_name])

        print(asr)
        print(res)
        # print('Avg ASR: {:.1f}'.format(sum(asr.values()) / len(asr)))
        # save_path = os.join('./log', args.result_path)
        # save_path = os.path.join('./log', args.result_path)
        # with open(save_path, 'a') as f:
        #     f.write(args.output_dir + res + '\n')


        # reduce the duplicate elements and keep the order
        not_correct_list_name = list(dict.fromkeys(not_correct_list_name))
        not_correct_list_label = list(dict.fromkeys(not_correct_list_label))

        print(len(not_correct_list_name))
        print(len(not_correct_list_label))

        ## save these two lists into files
        import pickle
        # Save the list to a file
        with open('not_correct_list_name.pkl', 'wb') as f:
            pickle.dump(not_correct_list_name, f)

        with open('not_correct_list_label.pkl', 'wb') as f:
            pickle.dump(not_correct_list_label, f)

    return args.output_dir, args.eval

if __name__ == '__main__':
    # torch.cuda.set_device(0)
    # set_seed(2023)
    # import time
    # print('record time')
    # time_start = time.time()
    output_dir, eval = main()
    # time_end = time.time()
    # # save time data to results.log
    # if not eval:
    #     with open('results_time.log', 'a') as f:
    #         f.write(output_dir+ '\n')
    #         f.write('time cost: {:.2f} s'.format((time_end - time_start)) + '\n')
    #         ## min
    #         f.write('time cost: {:.2f} min'.format((time_end - time_start) / 60) + '\n')
    #         ## hour
    #         f.write('time cost: {:.2f} h'.format((time_end - time_start) / 3600) + '\n')


