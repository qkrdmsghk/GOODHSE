_base_ = ['../_base_/schedules/classification.py', '../_base_/default_runtime.py']

# transform
train_pipeline = [dict(type="SmileToGraph", keys=["input"]), dict(type='Collect', keys=['input', 'gt_label', 'group'])]
test_pipeline = [dict(type="SmileToGraph", keys=["input"]), dict(type='Collect', keys=['input', 'gt_label', 'group'])]

# dataset
dataset_type = "LBAPDataset"
ann_file = './data/DrugOOD/lbap_core_ic50_size.json'

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(split="train", type=dataset_type, ann_file=ann_file, pipeline=train_pipeline),
    ood_val=dict(split="ood_val",
                 type=dataset_type,
                 ann_file=ann_file,
                 pipeline=test_pipeline,
                 rule="greater",
                 save_best="accuracy"),
    iid_val=dict(
        split="iid_val",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
    ),
    ood_test=dict(
        split="ood_test",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
    ),
    iid_test=dict(
        split="iid_test",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
    ),
    num_classes=2
)

num_class = 2
in_channels = 39
edge_dim = 10
JK = 'last'
residual = False
virtual_node = False
gnn_type = 'gin'
graph_pooling = 'sum'
pred_head = 'cls'

emb_dim = 128

decomp_dropout = 0.1
decomp_layers = 4
model_dropout = 0.5
model_layers = 4


decomp_model = dict(
    le_gnn=dict(
        type='LeGNN',
        in_channels=in_channels,
        hid_channels=emb_dim,
        num_layer=decomp_layers,
        drop_ratio=decomp_dropout,
        edge_dim=edge_dim
    ),
    node_virtual=dict(
        type='GNN_node_Virtualnode',
        num_layer=decomp_layers,
        emb_dim=emb_dim,
        input_dim=in_channels,
        JK=JK,
        residual=residual,
        gnn_type=gnn_type,
        edge_dim=edge_dim
    ),
    node=dict(
        type='GNN_node',
        num_layer=decomp_layers,
        emb_dim=emb_dim,
        input_dim=in_channels,
        drop_ratio=decomp_dropout,
        JK=JK,
        residual=residual,
        gnn_type=gnn_type,
        edge_dim=edge_dim
    )
)

model = dict(
    classifier=dict(
        type='GNN_classifier',        
        num_class=num_class,
        num_layer=model_layers,
        emb_dim=emb_dim,
        input_dim=in_channels,
        gnn_type=gnn_type,
        virtual_node=virtual_node,
        residual=residual,
        drop_ratio=model_dropout,
        JK=JK,
        graph_pooling=graph_pooling,
        pred_head=pred_head,
        edge_dim=edge_dim
    ),
    domain=dict(
        type='GNN_classifier',        
        num_class=num_class,
        num_layer=model_layers,
        emb_dim=emb_dim,
        input_dim=in_channels,
        gnn_type=gnn_type,
        virtual_node=virtual_node,
        residual=residual,
        drop_ratio=model_dropout,
        JK=JK,
        graph_pooling=graph_pooling,
        pred_head=pred_head,
        edge_dim=edge_dim
    )
)
