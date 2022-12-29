#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn
from PIL.Image import Resampling

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch, default=0')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training, default=200')
parser.add_argument('--batchSize', type=int, default=6, help='size of the batches, default=6')
parser.add_argument('--dataroot', type=str, default='datasets/claire/', help='root directory of the dataset, default=datasets/claire')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate, default=0.0002')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0, default=100')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed), default=128')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data, default=3')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data, default=3')
parser.add_argument('--cuda', action='store_true', help='use GPU computation, default=store_true')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation, default=8')

# MODIFY START
# parser.add_argument('--hasNet', type=bool, default='False', help='If there is already network to train')
# parser.add_argument('--generator_A2B', type=str, default='output/399_netG_A2B.pth', help='A2B generator checkpoint file, default=output/399_netG_A2B.pth')
# parser.add_argument('--generator_B2A', type=str, default='output/399_netG_B2A.pth', help='B2A generator checkpoint file, default=output/399_netG_B2A.pth')
# parser.add_argument('--netD_A', type=str, default='output/399_netD_A.pth', help='A Discriminator checkpoint file, default=output/399_netD_A.pth')
# parser.add_argument('--netD_B', type=str, default='output/399_netD_B.pth', help='B Discriminator checkpoint file, default=output/399_netD_B.pth')
# MODIFY END
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

# MODIFY START
# Load state dicts
# if opt.hasNet: ## TODO, bug fix
#     print("hasNET!!")
#     if opt.cuda:
#         netG_A2B.cuda() #netG_A2B.to(torch.device('cuda')) new
#         netG_B2A.cuda() #netG_B2A.to(torch.device('cuda'))
#         netG_A2B.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.generator_A2B).items()})
#         netG_B2A.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.generator_B2A).items()})
#         netD_A.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.netD_A).items()})
#         netD_B.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.netD_B).items()})
        
#     else:
#         netG_A2B.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.generator_A2B, map_location=torch.device('cpu')).items()})
#         netG_B2A.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.generator_B2A, map_location=torch.device('cpu')).items()})
#         netD_A.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.netD_A, map_location=torch.device('cpu')).items()})
#         netD_B.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.netD_B, map_location=torch.device('cpu')).items()})
#     netG_A2B.eval()
#     netG_B2A.eval()
#     netD_A.eval()
#     netD_B.eval()
# MODIFY END

if opt.cuda:
    #netG_A2B.cuda()
    netG_A2B.to(torch.device('cuda'))
    #netG_B2A.cuda()
    netG_B2A.to(torch.device('cuda'))
    #netD_A.cuda()
    netD_A.to(torch.device('cuda'))
    #netD_B.cuda()
    netD_B.to(torch.device('cuda'))

    netG_A2B = nn.DataParallel(netG_A2B, device_ids=[0])
    netG_B2A = nn.DataParallel(netG_B2A, device_ids=[0])
    netD_A = nn.DataParallel(netD_A, device_ids=[0])
    netD_B = nn.DataParallel(netD_B, device_ids=[0])

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
# target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
target_real = Tensor(opt.batchSize).fill_(1.0)
target_fake = Tensor(opt.batchSize).fill_(0.0)


# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Resampling.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # 歸一化到[0, 1] 維度轉换, 例如[128, 128, 1] --> [1, 128, 128]
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ] # FIX 將[0, 1]歸一化到[-1, 1]  mean, std
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):  
        # Set model input
        if(input_B.shape != batch['B'].shape or input_A.shape != batch['A'].shape):
            continue
        real_A = input_A.copy_(batch['A'])
        real_B = input_B.copy_(batch['B'])
        # print(real_A.shape)
        # print(real_B.shape)
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # same_B = netG_A2B(real_B)
        # loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # # G_B2A(A) should equal A if real A is fed
        # same_A = netG_B2A(real_A)
        # loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real) # generator讓pred_fake接近1

        # print("1")
        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # print("2")
        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        # print("3")
        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        #loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB


        loss_G.backward(retain_graph=True)
        optimizer_G.step()

        # loss_G.backward()
        # optimizer_G.step()

        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach()) # fake_A 由 G 生成, detach 使更新不影響G
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
        #             images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    if epoch % 100 == 99 or epoch == (opt.n_epochs) - 1:

        torch.save(netG_A2B.state_dict(), 'output/{}_netG_A2B.pth'.format(epoch))
        torch.save(netG_B2A.state_dict(), 'output/{}_netG_B2A.pth'.format(epoch))
        torch.save(netD_A.state_dict(), 'output/{}_netD_A.pth'.format(epoch))
        torch.save(netD_B.state_dict(), 'output/{}_netD_B.pth'.format(epoch))

###################################
