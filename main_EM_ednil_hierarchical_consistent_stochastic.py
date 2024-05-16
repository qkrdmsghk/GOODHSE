import argparse
import os
from datetime import datetime
from drugood.models import build_backbone
import torch
from drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator
from torch_geometric.data import DataLoader
from datasets.drugood_dataset import DrugOOD
from models.EM_models import MainModel, DomainHierarchyClassifier
from models.EM_Trainer_hierarchical import EM_EDNIL_Trainer_EI_hier, EM_EDNIL_Trainer_IL_hier
from utils.logger import Logger
from utils.util import args_print, set_seed
import pandas as pd
import wandb

def build_models_from_cfg(args, cfg, device, num_domains):
    main_model = MainModel(args, num_class=cfg.num_class, input_dim=39).to(device)
    domain_classifier = DomainHierarchyClassifier(args, num_task=1, num_domain=num_domains, num_class=cfg.num_class).to(device)
    return main_model, domain_classifier


def return_configs(args):
    cfg = Config.fromfile(os.path.join("configs", "ednil", args.dataset + ".py"))

    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = args.num_workers

    cfg.emb_dim = args.emb_dim
    cfg.decomp_dropout = args.dropout
    cfg.model_dropout = args.dropout
    cfg.model_layers = args.IL_num_layers
    cfg.decomp_layers = args.EI_num_layers
    
    cfg.decomp_model.node.num_layer = args.EI_num_layers
    cfg.model.classifier.num_layer = cfg.model.domain.num_layer = args.IL_num_layers
    cfg.decomp_model.node.emb_dim = cfg.model.classifier.emb_dim = cfg.model.domain.emb_dim = args.emb_dim
    cfg.decomp_model.node.drop_ratio = cfg.model.classifier.drop_ratio = cfg.model.domain.drop_ratio = args.dropout
    
    return cfg




def main():
    parser = argparse.ArgumentParser(description='Causality Inspired Invariant Graph LeArning')
    parser.add_argument('--wbproject_name', default='tuning', type=str, help='wandb project name')
    parser.add_argument('--device', default=1, type=int, help='cuda device')
    parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')
    parser.add_argument('--dataset', default='drugood_lbap_core_ic50_assay', type=str)

    # training config
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--EI_lr', default=1e-3, type=float, help='learning rate for the EI')
    parser.add_argument('--IL_lr', default=1e-4, type=float, help='learning rate for the IL')
    parser.add_argument('--seed', nargs='?', default='[1,2,3,4,5]', help='random seed')
    parser.add_argument('--pretrain', default=20, type=int, help='pretrain epoch before early stopping')

    # model config
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--r', default=0.8, type=float, help='selected ratio')
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='sum', type=str)
    parser.add_argument('--EI_num_layers', default=1, type=int)
    parser.add_argument('--IL_num_layers', default=4, type=int)
    parser.add_argument('--alpha', default=1, type=float, help='envConWeight')
    parser.add_argument('--beta', default=1, type=float, help='labelConWeight')
    

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--early_stopping', default=20, type=int) # 20, 5
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric', default='auc', type=str, help='specify a particular eval metric, e.g., mat for MatthewsCoef')

    # Invariant Learning baselines config
    parser.add_argument('--num_envs', default=-1, type=int, help='num of envs need to be partitioned')
    parser.add_argument('--irm_p', default=0.01, type=float, help='penalty weight')
    parser.add_argument('--irm_opt', default='ednil_EI_hier_assay', type=str, help='algorithms to use')
    
    # EDNIL config
    parser.add_argument('--EI_epochs', default=10, type=int)  # epochs for EI
    parser.add_argument('--IL_epochs', default=400, type=int)  # epochs for IL
    parser.add_argument('--temperature', default=0.2, type=float)  # temperature for ednil
    parser.add_argument('--envw_thres', default=2, type=float)  # threshold for env weight
    parser.add_argument('--penalty_w', default=-1, type=float)  # penalty weight for env weight
    parser.add_argument('--l2_w', default=-1, type=float)  # l2 weight for env weight
    
    
    parser.add_argument('--num_hierarchy', default=3, type=int)  # number of hierarchy
    parser.add_argument('--ei_last_hierarchy', default=2, type=int)  # number of last hierarchy for EI
    parser.add_argument('--il_last_hierarchy', action='store_true', default=True)  # whether to use only last hiearachy in IL
    parser.add_argument('--il_cls', default='linear', type=str)  # classification type in IL


    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--save_model', action='store_true')  # save pred to ./pred if not empty
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # erm_model = None  # used to obtain pesudo labels for CNC sampling in contrastive loss

    args.seed = eval(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    if args.dataset.lower().startswith('drugood'):
        # drugood_lbap_core_ic50_assay.json
        cfg = return_configs(args)        
        root = os.path.join(args.root, "DrugOOD")
        train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args.dataset, mode="train")
        val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args.dataset, mode="ood_val")
        test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args.dataset, mode="ood_test")
        if args.eval_metric == 'auc':
            args.evaluator = Evaluator('ogbg-molhiv')
            args.eval_metric = 'rocauc'
        else:
            args.evaluator = Evaluator('ogbg-ppa')
        args.edge_dim=10
        args.input_dim=39
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = args.commit
    if not os.path.exists(os.path.join('./logs', args.irm_opt)):
        os.mkdir(os.path.join('./logs', args.irm_opt))
    exp_dir = os.path.join('./logs', args.irm_opt, experiment_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + f'/log_{datetime_now[4::]}.log')
    args_print(args, logger)

    logger.info(f"# Train: {len(train_loader.dataset)}  #Val: {len(valid_loader.dataset)} #Test: {len(test_loader.dataset)} ")
    best_weights = None
    
    # generate environment partitions ==> predefined environment!
    if args.num_envs == -1:
        env_idx = []
        for graph in train_loader.dataset:
            env_idx.append(graph.group)
        env_idx = torch.cat(env_idx, dim=0)
        num_envs = len(set(env_idx.tolist()))
        print(f"num of envs: {num_envs}")
        num_envs = [num_envs]
    else:
        num_envs = [args.num_envs]
    if 'hier' in args.irm_opt:
        print(f'[INFO] Using the hierarchical model...')
        if args.num_hierarchy > 2:
            for i in range(args.num_hierarchy-2):
                num_envs.append(num_envs[-1]//2)
            num_envs.append(args.ei_last_hierarchy)
        elif args.num_hierarchy == 2:
            num_envs.append(args.ei_last_hierarchy)
        elif args.num_hierarchy == 1:
            num_envs = num_envs
        print(f"num of envs: {num_envs}")

    def make_log(args, seed):
        log_dir = os.path.join('wandb_log', args.dataset, args.irm_opt)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fname = args.commit + '_seed_' + str(seed) + '.json'
        return log_dir, fname

    log_test_perf = []

    for seed in args.seed:
        log_dir, log_name = make_log(args, seed)
        args.seed_i = seed

        wandb.init(config=args, 
        project=args.wbproject_name,
        name=log_name,
        dir=log_dir,
        reinit=True)
        # else:
        #     wandb = None
        
        set_seed(seed)
        # models and optimizers
        il_model, ei_domain = build_models_from_cfg(args, cfg, device, num_envs)
        il_model_optimizer = torch.optim.Adam(list(il_model.parameters()), lr=args.IL_lr)
        ei_pre_optimizer = torch.optim.Adam(list(ei_domain.parameters()), lr=args.EI_lr)
        ei_domain_optimizer = torch.optim.Adam(list(ei_domain.parameters()), lr=args.EI_lr)

        if 'hier' in args.irm_opt:
            ei_trainer = EM_EDNIL_Trainer_EI_hier(
                num_classes=cfg.data.num_classes,
                model=ei_domain,
                pre_optimizer=ei_pre_optimizer,
                optimizer=ei_domain_optimizer,
                temperature=args.temperature,
                device=device)
            
            il_trainer = EM_EDNIL_Trainer_IL_hier(
                num_classes=cfg.data.num_classes,
                model=il_model,
                optimizer=il_model_optimizer,
                device=device,
                args=args)
        else:
            print(f'[ERROR INFO] Not using the hierarchical model...')
    
        env_model = None
    
        if not os.path.exists(os.path.join(exp_dir, 'ei_hier_seed{}.pt'.format(seed))) and args.EI_epochs > 0:
            print(f'[INFO] START training on assistant EI models')
            best_ep, min_loss = ei_trainer.train_EI_hier_consistent(train_loader, args, wandb=wandb)
            print('--best epoch: {}-- best ei_loss: {:.4f}'.format(best_ep, min_loss))
            torch.save(ei_trainer.model.state_dict(), os.path.join(exp_dir, 'ei_hier_seed{}.pt'.format(seed)))

        print(f'[INFO] Loading the pretrained EI model...')
        ei_trainer.model.load_state_dict(torch.load(os.path.join(exp_dir, 'ei_hier_seed{}.pt'.format(seed))))
        env_model = ei_trainer.model
        env_model.eval()

        print(f'[INFO] START training on main IL model')
        best_test_perf, best_val_perf, best_epoch = il_trainer.train_IL_hier(train_loader, valid_loader, test_loader, args, env_model, wandb=wandb)
        # torch.save(il_trainer.model.state_dict(), os.path.join(exp_dir, 'il_seed{}.pt'.format(seed)))

        # print('[INFO] EVALUATING the main model...')
        # test_perf = il_trainer.test(test_loader, args, env_model)
        print('[INFO] Last: Test_perf: {:.4f} Val_perf:{:.4f} '.format(best_test_perf, best_val_perf))
        logger.info("Best performance at Epoch: {}".format(best_epoch))
        logger.info("+" * 50)
        logger.info("Last: Test_perf: {:.4f} Val_perf:{:.4f} ".format(best_test_perf, best_val_perf))
        logger.info("=" * 50)
        log_test_perf.append(best_test_perf)
    
    result = pd.DataFrame(log_test_perf).T
    result.columns = [f'seed_{i}' for i in args.seed]
    result['mean'] = result.mean(axis=1)
    result['std'] = result.std(axis=1)
    result.to_csv(os.path.join(exp_dir, 'result.csv'), sep='\t', index=False)
    
    log_test_perf = torch.tensor(log_test_perf)
    logger.info("Mean: {} Std: {}".format(torch.mean(log_test_perf), torch.std(log_test_perf)))

    print("\n\n\n")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
