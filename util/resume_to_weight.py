# tranform a resume checkpoints file to a pure model weight file

import torch

def resume_to_weight(resume_path, weight_path):
    state = torch.load(resume_path, map_location='cpu')

    pretrained_dict = state['params']
    print("pretrained_dict: ", pretrained_dict.keys())

    pretrained_dict = {k.replace("backbone.module.", "encoder."): v for k, v in pretrained_dict.items()}

    state = {'params': pretrained_dict}

    torch.save(state, weight_path)

if __name__ == '__main__':
    resume_path = "../max_acc_sim.pth"
    weight_path = '../Vit_small_mae_pretrained.pth'

    resume_to_weight(resume_path, weight_path)