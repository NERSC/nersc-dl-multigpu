import os, sys, time
import argparse
import torch
import numpy as np
from utils.YParams import YParams
from utils.data_loader import get_data_loader
from ruamel.yaml import YAML
import torch.optim as optim
from torch.optim import lr_scheduler

# models
import models.cnn

class Trainer():
    """ trainer class """
    def __init__(self, params, args):
        self.root_dir = args.root_dir
        self.config = args.config 
        self.run_num = args.run_num

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
        
        self.log_to_screen = 1 # print to screen
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        self.params = params

    def init_exp_dir(self, exp_dir):
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
            os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False


    def launch(self):
        exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])
        self.init_exp_dir(exp_dir)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = self.params.valid_batch_size

        # get the dataloaders
        self.train_data_loader, self.train_sampler = get_data_loader(self.params, self.params.train_path, False, train=True)
        self.val_data_loader, self.valid_sampler = get_data_loader(self.params, self.params.val_path, False, train=False)

        # get the model
        self.model = models.cnn.simple_cnn(self.params).to(self.device)

        # set an optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.max_cosine_lr_epochs)

        # set loss functions
        self.loss_func = torch.nn.MSELoss()

        # checkpointing
        self.iters = 0
        self.startEpoch = 0
        if self.params.resuming:
            print("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)
        self.epoch = self.startEpoch
        self.logs = {}

        # launch training
        self.train()

    def train(self):
        if self.log_to_screen:
            print("Starting training loop...")

        best_loss = np.inf
        best_epoch = 0
        self.logs['best_epoch'] = best_epoch

        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch = epoch
            start = time.time()

            # training
            tr_time  = self.train_one_epoch()
            # validation
            val_time = self.val_one_epoch()

            # learning rate scheduler
            self.scheduler.step()

            # keep track of best model according to validation loss
            if self.logs['val_loss'] <= best_loss:
                is_best_loss = True
                best_loss = self.logs['val_loss']
            else:
                is_best_loss = False
            self.logs['best_val_loss'] = best_loss
            best_epoch = self.epoch if is_best_loss else best_epoch
            self.logs['best_epoch'] = best_epoch

            # save checkpoint (if best epoch additionally save the best epoch too)
            if self.params.save_checkpoint:
                #checkpoint at the end of every epoch
                self.save_checkpoint(self.params.checkpoint_path, is_best=is_best_loss)

            # some print statements
            if self.log_to_screen:
                print('Time taken for epoch {} is {} sec; with {}/{} in tr/val'.format(self.epoch+1, time.time()-start, tr_time, val_time))
                print('Loss = {}, Val loss = {}'.format(self.logs['train_loss'], self.logs['val_loss']))


    def train_one_epoch(self):
        tr_time = 0
        self.model.train()

        # buffers for logs
        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['train_loss'] = logs_buff[0].view(-1)

        for i, (inputs, targets) in enumerate(self.train_data_loader):
            self.iters += 1
            data_start = time.time()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            tr_start = time.time()

            self.model.zero_grad()
            u = self.model(inputs)

            loss = self.loss_func(u, targets)
            loss.backward()
            self.optimizer.step()
 
            # add all the minibatch losses
            self.logs['train_loss'] += loss.detach()

            tr_time += time.time() - tr_start

        self.logs['train_loss'] /= len(self.train_data_loader)

        return tr_time

    def val_one_epoch(self):
        self.model.eval()
        val_start = time.time()

        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['val_loss'] = logs_buff[0].view(-1)

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                u = self.model(inputs)
                loss = self.loss_func(u, targets)
                self.logs['val_loss'] += loss.detach()

        self.logs['val_loss'] /= len(self.val_data_loader)

        val_time = time.time() - val_start

        return val_time

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler is not None else None)}, checkpoint_path)
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if  self.scheduler is not None else None)}, checkpoint_path.replace('.tar', '_best.tar'))

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(0)) 
        self.model.load_state_dict(checkpoint['model_state'])
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

if __name__ == '__main__':
    # parsers for any cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./configs/default.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--root_dir", default='./outputs', type=str, help='root dir to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    trainer = Trainer(params, args)

    trainer.launch()

    print('Training complete')
