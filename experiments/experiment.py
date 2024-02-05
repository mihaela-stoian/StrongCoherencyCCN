import time

import torch
from pishield.propositional_constraints.constraints_layer import ConstraintsLayer
from pishield.propositional_constraints.util import draw_classes, train, test
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from shapes import ShapeDataset


class Experiment():
    def __init__(self, name, model, shapes, points = 2500, batch_size=2500):
        self.name = name
        self.model = model 
        self.shapes = shapes 

        # Build dataset
        train_data = ShapeDataset(shapes, points)
        test_data = ShapeDataset(shapes, points)

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # Build optimizer
        learning_rate = 1e-1
        self.optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999))
        self.loss_fn = nn.BCELoss()

    @classmethod
    def get_ratio(cls, progress, progressive):
        if progress > progressive:
            return 1.
        else:
            return progress / progressive

    @classmethod
    def get_ratios(cls, epochs, progressive):
        return [cls.get_ratio((t + 1) / epochs, progressive) for t in range(epochs)]

    def run(self, epochs, device='cpu', progressive=0.):
        self.model, self.clayer = self.model.to(device), self.clayer.to(device)

        ratios = Experiment.get_ratios(epochs, progressive)
        sw = SummaryWriter()

        for t, ratio in enumerate(ratios):
            if t % 10 == 0:
                print(f"Epoch {t + 1}/{len(ratios)}, Ratio {ratio}\n-----------------------")
            self.train_step(device, ratio=ratio)
            loss, correct = test(self.test_dataloader, self.model, self.clayer, self.loss_fn, device)

            sw.add_scalar('Loss/test', loss, t)
            for i, rate in enumerate(correct):
                sw.add_scalar(f'Accuracy/test (label {i})', rate, t)
            self.test_loss = loss

            if t % 1000 == 0:
                self.draw_results(sw=sw, epoch=t)

        print("Done!")
        self.draw_results(sw=sw, epoch=len(ratios))
        sw.close()

    def train_step(self):
        pass

    def experiment_path(self, dir):
        return f"{dir + self.name}-{self.test_loss:.5}-{int(time.time())}"

    def draw_results(self, path=None, sw=None, epoch=0):
        path1 = path + '.png' if path != None else None 
        path2 = path + "-constrained.png" if path != None else None

        full = False
        
        prev = draw_classes(self.model, draw=(lambda ax, i: self.shapes[i].plot(ax, full=full)), path=path1)
        after = draw_classes(nn.Sequential(self.model, self.clayer), draw=(lambda ax, i: self.shapes[i].plot(ax, full=full)), path=path2)
        if not sw is None:
            sw.add_figure('previous', prev, epoch)
            sw.add_figure('after', after, epoch)


    def save(self, dir='./'):
        path = self.experiment_path(dir)
        torch.save(self.model, path + ".pth")
        torch.save(self.clayer, path + "_clayer.pth")
        print('Saved experiment at {path}'.format(path=path))
        self.draw_results(path=path)


def load_trained_model(path):
    model = torch.load(path + '.pth')
    clayer = torch.load(path + '_clayer.pth')
    return model, clayer

