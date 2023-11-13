import torch.nn as nn
import torch
from utils.global_vars import dtype

bce_loss = nn.BCELoss().type(dtype)
mse_loss = nn.MSELoss().type(dtype)
l1_loss = nn.L1Loss()

class Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, epsilon=1e-3, avg=True):
        super(Charbonnier_loss, self).__init__()
        self.eps = epsilon ** 2
        # self.eps = Variable(torch.from_numpy(np.asarray([epsilon ** 2])))
        # self.eps = Variable(torch.ones())
        self.avg = avg


    def forward(self, X, Y, mask=None):
        batchsize = X.data.shape[0]
        diff = X - Y
        square_err = diff ** 2
        if not mask is None:
            square_err = square_err * mask
        square_err_sum_list = torch.sum(square_err, dim=1)
        square_err_sum_list = torch.sum(square_err_sum_list, dim=1)
        square_err_sum_list = torch.sum(square_err_sum_list, dim=1)
        square_err_sum_list = square_err_sum_list + self.eps
        error = torch.sqrt(square_err_sum_list)
        if self.avg:
            loss = torch.sum(error) / batchsize
        else:
            loss = error
        return loss


class Recall_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, threshold=None, epsilon=1e-5, avg=True):
        super(Recall_loss, self).__init__()
        self.eps = epsilon
        self.avg = avg
        self.threshold = threshold

    def forward(self, X, Y):
        X = torch.clamp(X, min=0, max=1)
        batchsize = X.data.shape[0]
        if not self.threshold is None:
            X = X > self.threshold

        intersection = X * Y
        intersection = intersection.reshape(batchsize, -1)
        intersection = torch.sum(intersection, dim=1)
        Y_pos = torch.sum(Y.reshape(batchsize, -1), dim=1)
        recall = (intersection + self.eps) / (Y_pos + self.eps)
        if self.avg:
            loss = torch.sum(recall) / batchsize
        else:
            loss = recall
        return loss


class Precision_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, threshold=None, epsilon=1e-5, avg=True):
        super(Precision_loss, self).__init__()
        self.eps = epsilon
        self.avg = avg
        self.threshold = threshold

    def forward(self, X, Y):
        X = torch.clamp(X, min=0, max=1)
        batchsize = X.data.shape[0]
        if not self.threshold is None:
            X = X > self.threshold

        intersection = X * Y
        intersection = intersection.reshape(batchsize, -1)
        intersection = torch.sum(intersection, dim=1)
        X_pos = torch.sum(X.reshape(batchsize, -1), dim=1)
        prec = (intersection + self.eps) / (X_pos + self.eps)
        if self.avg:
            loss = torch.sum(prec) / batchsize
        else:
            loss = prec
        return loss

class Recall_Guided_RegLoss(nn.Module):
    def __init__(self, threshold=None, gamma=1):
        super(Recall_Guided_RegLoss, self).__init__()
        self.recall_loss = Recall_loss(threshold=threshold, avg=False)
        self.c_loss = Charbonnier_loss(avg=False)
        self.gamma = gamma

    def forward(self, X, Y, recall_grad=False):
        if recall_grad:
            recall = self.recall_loss(X, Y)
        else:
            with torch.no_grad():
                recall = self.recall_loss(X, Y)
        loss = self.c_loss(X, Y, mask=Y) # Calculate foreground loss
        # loss = torch.mean((1 - recall) * loss)
        loss = torch.mean((1 - recall) ** self.gamma * loss)

        return loss, torch.mean(recall)

class Precision_Guided_RegLoss(nn.Module):
    def __init__(self, threshold=None, gamma=1):
        super(Precision_Guided_RegLoss, self).__init__()
        self.prec_loss = Precision_loss(threshold=threshold, avg=False)
        self.c_loss = Charbonnier_loss(avg=False)
        self.gamma = gamma

    def forward(self, X, Y, recall_grad=False):
        if recall_grad:
            precision = self.prec_loss(X, Y)
        else:
            with torch.no_grad():
                precision = self.prec_loss(X, Y)
        loss = self.c_loss(X, Y, mask=1-Y) # Calculate background loss
        # loss = torch.mean((1 - precision) * loss)
        loss = torch.mean((1 - precision) ** self.gamma * loss)

        return loss, torch.mean(precision)


def C_Loss(output, label, mask=None):
    c_loss_func = Charbonnier_loss(epsilon=1e-3)
    return c_loss_func(output, label, mask)
