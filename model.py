import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

class MLC(nn.Module):

    def __init__(self,queue_size , data_size):
        super().__init__()

        # create the queue
        self.register_buffer("queue", torch.randn(queue_size,data_size))
        self.register_buffer("queue_pseudo", torch.randn(queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        # gather keys before updating queue
        keys = keys
        labels = labels

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self,main_net, main_opt, hard_loss_f,
                    meta_net, meta_opt, soft_loss_f,
                    data_s, target_s, data_g, target_g,
                    eta, args, vnet, vnet1, soft_label):

        # compute gw for updating meta_net

        logit_g = main_net(data_g)
        loss_gg = hard_loss_f(logit_g, target_g)
        loss_gg = loss_gg.unsqueeze(0)
        l_lambda = vnet(loss_gg)
        loss1 = hard_loss_f(logit_g, target_g)
        loss1 = loss1.unsqueeze(0)
        loss_g1 = torch.reshape(loss1, (len(loss1), 1))
        l1 = torch.sum(loss_g1 * l_lambda) / len(loss_g1)
        loss_g = loss1 + l1
        gw = torch.autograd.grad(loss_g, main_net.parameters())

        # given current meta net, get corrected label
        logit_s, x_s_h = main_net(data_s, return_h=True)
        pseudo_target_s = meta_net(x_s_h.detach(), target_s)

        loss_s, new_label = soft_loss_f(logit_s, target_s, pseudo_target_s)

        f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)

        f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True)
        # 2. set w as w'
        f_param = []
        for i, param in enumerate(main_net.parameters()):
            f_param.append(param.data.clone())
            param.data = f_params_new[i].data  # use data only as f_params_new has graph

        # training loss Hessian approximation
        Hw = 1  # assume to be identity

        # 3. compute d_w' L_{D}(w')
        logit_g = main_net(data_g)
        loss_g2 = hard_loss_f(logit_g, target_g)
        loss_g2 = loss_g2.unsqueeze(0)
        l_lambda = vnet(loss_g2)
        loss12 = hard_loss_f(logit_g, target_g)
        loss12 = loss12.unsqueeze(0)
        loss_g1 = torch.reshape(loss12, (len(loss12), 1))
        l11 = torch.sum(loss_g1 * l_lambda) / len(loss_g1)
        loss_g = loss12 + l11

        gw_prime = torch.autograd.grad(loss_g, main_net.parameters())

        # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2
        tmp1 = [(1 - Hw * dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
        gw_norm2 = (_concat(gw).norm()) ** 2
        tmp2 = [gw[i] / gw_norm2 for i in range(len(gw))]
        gamma = torch.dot(_concat(tmp1), _concat(tmp2))

        # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
        Lgw_prime = [dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]

        proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))

        # back prop on alphas
        meta_opt.zero_grad()
        proxy_g.backward()

        # accumulate discounted iterative gradient
        for i, param in enumerate(meta_net.parameters()):
            if param.grad is not None:
                param.grad.add_(gamma * args.dw_prev[i])
                args.dw_prev[i] = param.grad.clone()

        if (args.steps + 1) % (args.gradient_steps) == 0:  # T steps proceeded by main_net
            meta_opt.step()
            args.dw_prev = [0 for param in meta_net.parameters()]  # 0 to reset

        # modify to w, and then do actual update main_net
        for i, param in enumerate(main_net.parameters()):
            param.data = f_param[i]
            param.grad = f_param_grads[i].data
        main_opt.step()

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss_s)
        prob = gmm.predict_proba(loss_s)
        prob = prob[:, gmm.means_.argmin()]

        self._dequeue_and_enqueue(k, pseudo_labels_b)


        return loss_g, loss_s, new_label,self.queue,self.queue_pseudo


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

@torch.no_grad()
def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')

# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov']

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay # s=1
        s = 1

        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. -dampening) # s=1.-dampening

            if nesterov:
                dparam = dparam + momentum * moment # s= 1+momentum*(1.-dampening)
                s = 1 + momentum*(1.-dampening)
            else:
                dparam = moment # s=1.-dampening
                s = 1.-dampening

        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam  * eta)

        if return_s:
            ss.append(s*eta)

    if return_s:
        return ans, ss
    else:
        return ans