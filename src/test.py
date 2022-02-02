import torch
from modules.retrieval.model import TIRGModel
# import torch
from utils.arguments import load_edict_config
import torch.nn as nn
import torch.optim as optim
from modules.propv1.model_scain import GeNeVAPropV1ScainModel
from collections import namedtuple

WEIGHT_PATH = "./results/experiments/exp78-pretrain-tirg/snapshots/codraw_pretrain_models_epoch_10.pth"
PATH = "./params/exp169.yaml"

Args = namedtuple('args','yaml_path')
cfg = load_edict_config(Args(PATH))

if "use_stap_disc" not in cfg:
    cfg.use_stap_disc = False
if "use_relnet" not in cfg:
    cfg.use_relnet = False
if "use_gate_for_stap" not in cfg:
    cfg.use_gate_for_stap = False
if "use_co_attention" not in cfg:
    cfg.use_co_attention = False
if "discriminator_arch" not in cfg:
    cfg.discriminator_arch = "standard"


model = TIRGModel(
            depth=cfg.depth,
            sentence_encoder_type=cfg.sentence_encoder_type,
            with_spade=cfg.with_spade,
            text_dim=cfg.text_dim,
            hidden_dim=cfg.hidden_dim,
            sa_fused=cfg.sa_fused,
            sta_concat=cfg.sta_concat,
            use_pos_emb=cfg.use_pos_emb,
            sa_gate=cfg.sa_gate,
            res_mask=cfg.res_mask,
            res_mask_post=cfg.res_mask_post,
            use_conv_final=cfg.use_conv_final,
            multi_channel_gate=cfg.multi_channel_gate,
            optimizer_type=cfg.optimizer_type,
            lr=cfg.lr,
            lr_bert=cfg.lr_bert if "lr_bert" in cfg.keys() else 2e-5,
            weight_decay=cfg.weight_decay,
            loss_type=cfg.loss_type,
            margin=cfg.margin,
            k=cfg.k,
            gate_loss_gamma=cfg.gate_loss_gamma,
            text_detection_gamma=cfg.text_detection_gamma,
            gate_detection_gamma=cfg.gate_detection_gamma,
            num_objects=cfg.num_objects,
        )
# Load
model = GeNeVAPropV1ScainModel(
            # generator
            image_feat_dim=cfg.image_feat_dim,
            generator_sn=cfg.generator_sn,
            generator_norm=cfg.generator_norm,
            embedding_dim=cfg.embedding_dim,
            condaug_out_dim=cfg.condaug_out_dim,
            cond_kl_reg=cfg.cond_kl_reg,
            noise_dim=cfg.noise_dim,
            gen_fusion=cfg.gen_fusion,
            sta=cfg.sta,
            nhead=cfg.nhead,
            res_mask_post=cfg.res_mask_post,
            multi_channel_gate=cfg.multi_channel_gate,
            use_relnet=cfg.use_relnet,
            # discriminator
            discriminator_arch=cfg.discriminator_arch,
            discriminator_sn=cfg.discriminator_sn,
            num_objects=cfg.num_objects,
            disc_fusion=cfg.disc_fusion,
            use_stap_disc=cfg.use_stap_disc,
            use_gate_for_stap=cfg.use_gate_for_stap,
            use_co_attention=cfg.use_co_attention,
            # misc
            generator_lr=cfg.generator_lr,
            generator_beta1=cfg.generator_beta1,
            generator_beta2=cfg.generator_beta2,
            discriminator_lr=cfg.discriminator_lr,
            discriminator_beta1=cfg.discriminator_beta1,
            discriminator_beta2=cfg.discriminator_beta2,
            wrong_fake_ratio=cfg.wrong_fake_ratio,
            aux_reg=cfg.aux_reg,
            gp_reg=cfg.gp_reg,
        )

checkpoint: dict = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))

# print(checkpoint.keys())

import pprint

printer = pprint.PrettyPrinter(indent=4)

printer.pprint(list(checkpoint.keys()))

# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()