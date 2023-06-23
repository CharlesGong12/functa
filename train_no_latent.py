import os
import argparse
import sys
# sys.path.append('/home/aistudio/work/libraries')
# import wandb

import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F


import model
import utils
from dataloaders import create_dataloader



def outer_step_no_latent(images, coords, network, optim, criterion):
    recon = network(coords)

    loss = criterion(recon, images)
    loss = loss.mean([1, 2, 3]).sum(0)

    loss.backward()
    optim.step()
    optim.clear_grad()
    network.clear_gradients()

    return loss.numpy()


def parse_args():
    """
    command args
    """
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--dset", dest="dset", help="Use dataset", default='cifar10', type=str)

    parser.add_argument(
        "--dsetRoot", dest="dsetRoot", help="Rootpath of dataset", default=None, type=str)

    parser.add_argument(
        "--output", dest="output", help="Rootpath of output", default="./output/", type=str)

    parser.add_argument(
        "--batchsize", dest="batchsize", default=16, type=int)
    # parser.add_argument("--wandb_id",default='functa2' ,type=str)

    return parser.parse_args()


if __name__ == "__main__":
    PLACE = paddle.CUDAPlace(0)
    # print(paddle.device.get_device())
    paddle.device.set_device('gpu:0')
    # Args
    args = parse_args()
    # Config
    batch_size = args.batchsize
    inner_steps = 3
    outer_steps = 100000
    inner_lr = 5e-3
    outer_lr = 1e-6
    latent_init_scale = 0.01
    save_interval = 100 # save ckpt interval
    ckpt_dir = args.output

    # Dataloader
    dataloader = create_dataloader.create_dataloader(args.dset, args.dsetRoot, batch_size)

    # print('loaded')
    # Prepare
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)


    # wandb.init(
    #             project = "nndl",
    #             entity='gongcaho',
    #             config = args,
    #             name = 'debug2',
    #             id = args.wandb_id,
    #             #resume = True,
    #             )


    # Model
    model_cfg = {
        'out_channels': 3, # RGB
        'depth': 15,
        'meta_sgd_clip_range': [0, 1],
        'meta_sgd_init_range': [0.005, 0.1],
        'modulate_scale': False,
        'modulate_shift': True,
        'use_meta_sgd': True,
        'w0': 30,
        'width': 1024}

    if args.dset == "mnist":
        model_cfg['out_channels'] = 1

    network = model.ModulatedSiren(**model_cfg)
    # network.set_state_dict(paddle.load("/home/aistudio/output/noLatent_cifar10_1024.pdparams"))

    # Optimizer
    ## Inner optimizer

    ## Outer optimizer
    outer_optim = paddle.optimizer.Adam(outer_lr, weight_decay=1e-4,
                    parameters=network.parameters())

    # Loss
    criterion = nn.MSELoss(reduction='none')

    # Multi GPU prepare
    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True, strategy=utils.get_strategy())
        outer_optim = paddle.distributed.fleet.distributed_optimizer(outer_optim)
        ddp_network = paddle.DataParallel(network, find_unused_parameters=True)

    # Train loop
    iter = 0
    iterator = dataloader.__iter__()
    #print('Begin training')

    while iter < outer_steps:
        try:
            images, coords, labels, idxs = iterator.next()
        except StopIteration:
            iterator = dataloader.__iter__()
            images, coords, labels, idxs = iterator.next()
        #print('data loaded')
        iter += 1
        if iter > outer_steps: break


        outer_loss = outer_step_no_latent(images, coords, ddp_network if nranks > 1 else network, outer_optim, criterion)
        if local_rank == 0 and iter%200==0:
            psnr = -10 * np.log10(outer_loss / batch_size)
            print("Outer iter {}/{}: outer loss {:.6f}, outer PSNR {:.6f}".format(iter, 
                    outer_steps, outer_loss[0], psnr[0]))
            # wandb.log({'outer loss': outer_loss[0],
            #     'outer psnr': psnr[0],
            #     'epoch':iter ,
            #     })

        if local_rank == 0 and iter > 0 and iter % save_interval == 0:
            current_save_dir = ckpt_dir
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(network.state_dict(), os.path.join(current_save_dir, 'noLatent_{}_{}.pdparams'.format(args.dset,model_cfg['width'])))

    iterator.__del__()