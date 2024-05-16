import torch
import numpy as np

import random
from texttable import Texttable
from sklearn.metrics import matthews_corrcoef
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_partition(len_dataset, device, seed, p=[0.5,0.5]):
    '''
        group the graph randomly

        Input:   len_dataset   -> [int]
                 the number of data to be groupped
                 
                 device        -> [torch.device]
                
                 p             -> [list]
                 probabilities of the random assignment for each group
        Output: 
                 vec           -> [torch.LongTensor]
                 group assignment for each data
    '''
    assert abs(np.sum(p) - 1) < 1e-4
    
    vec = torch.tensor([]).to(device)
    for idx, idx_p in enumerate(p):
        vec = torch.cat([vec, torch.ones(int(len_dataset * idx_p)).to(device) * idx])
        
    vec = torch.cat([vec, torch.ones(len_dataset - len(vec)).to(device) * idx])
    perm = torch.randperm(len_dataset, generator=torch.Generator().manual_seed(seed))
    return vec.long()[perm]

    
def args_print(args, logger):
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, str(_dict[k])])
    logger.info(table.draw())
    

def PrintGraph(graph):

    if graph.name:
        print("Name: %s" % graph.name)
    print("# Nodes:%6d      | # Edges:%6d |  Class: %2d" \
          % (graph.num_nodes, graph.num_edges, graph.y))

    print("# Node features: %3d| # Edge feature(s): %3d" \
          % (graph.num_node_features, graph.num_edge_features))


@torch.no_grad()
def eval_model(model, device, loader, evaluator, env_model=None, eval_metric='acc', save_pred=False):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                preds = model(batch, env_model=env_model)
                pred = preds[0]
            is_labeled = batch.y == batch.y
            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat':
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                if is_labeled.size() != pred.size():
                    with torch.no_grad():
                        preds, rep = model(batch, return_data="rep", debug=True)
                        pred = preds[0]
                        print(rep.size())
                    print(batch)
                    print(global_mean_pool(batch.x, batch.batch).size())
                    print(pred.shape)
                    print(batch.y.size())
                    print(sum(is_labeled))
                    print(batch.y)
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if eval_metric == 'mat':
        res_metric = matthews_corrcoef(y_true, y_pred)
    else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res_metric = evaluator.eval(input_dict)[eval_metric]

    if save_pred:
        return res_metric, y_pred
    else:
        return res_metric
