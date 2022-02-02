import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast

from data.codraw_retrieval_dataset import CoDrawRetrievalDataset, codraw_retrieval_collate_fn
from data.iclevr_retrieval_dataset import ICLEVRRetrievalDataset, iclevr_retrieval_collate_fn
from modules.retrieval.sentence_encoder import SentenceEncoder, BERTSentenceEncoder
from scripts.embs_spawn import run_dataloader
from tpu_device import devices, tpu_device
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from logging import getLogger
logger = getLogger(__name__)


class SentenceEmbeddingGenerator:
    def __init__(self, cfg):
        self.cfg = cfg

        # model
        self.sentence_encoder_type = cfg.sentence_encoder_type
        if "model_path" not in cfg:
            logger.warning(
                "model_path is not specified. "
                "use initial weight of pretrained models."
            )
            state_dict = None
        else:
            state_dict = torch.load(
                cfg.model_path, map_location=torch.device('cpu'))

        if cfg.sentence_encoder_type == "rnn":
            self.rnn_prev = nn.DataParallel(
                SentenceEncoder(cfg.text_dim),
                device_ids=devices,
            )

            self.rnn_curr = nn.DataParallel(
                SentenceEncoder(cfg.text_dim),
                device_ids=devices,
            )

            if state_dict is not None:
                self.rnn_prev.load_state_dict(state_dict["rnn_prev"])
                self.rnn_curr.load_state_dict(state_dict["rnn_curr"])

            self.rnn_prev = self.rnn_prev.to(tpu_device)
            self.rnn_curr = self.rnn_curr.to(tpu_device)

            self.rnn_prev.eval()
            self.rnn_curr.eval()
        elif cfg.sentence_encoder_type == "bert":
            self.tokenizer = BertTokenizerFast.\
                from_pretrained("bert-base-uncased")
            self.bert = nn.DataParallel(
                BERTSentenceEncoder(),
                device_ids=devices,
            ).to(tpu_device)
            if state_dict is not None:
                self.bert.load_state_dict(state_dict["bert"])
            self.bert.eval()
        else:
            raise ValueError

        # dataset
        if cfg.dataset == "codraw":
            # codraw-train
            self.dataset = CoDrawRetrievalDataset(
                cfg.dataset_path,
                cfg.glove_path,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.dataloader.collate_fn = codraw_retrieval_collate_fn
            # codraw-valid
            self.valid_dataset = CoDrawRetrievalDataset(
                cfg.valid_dataset_path,
                cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.valid_dataloader.collate_fn = codraw_retrieval_collate_fn
            # codraw-test
            self.test_dataset = CoDrawRetrievalDataset(
                cfg.test_dataset_path,
                cfg.glove_path,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.test_dataloader.collate_fn = codraw_retrieval_collate_fn
        elif cfg.dataset == "iclevr":
            # iclevr-train
            self.dataset = ICLEVRRetrievalDataset(
                cfg.dataset_path,
                cfg.glove_path,
            )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.dataloader.collate_fn = iclevr_retrieval_collate_fn
            # iclevr-valid
            self.valid_dataset = ICLEVRRetrievalDataset(
                cfg.valid_dataset_path,
                cfg.glove_path,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.valid_dataloader.collate_fn = iclevr_retrieval_collate_fn
            # iclevr-test
            self.test_dataset = ICLEVRRetrievalDataset(
                cfg.test_dataset_path,
                cfg.glove_path,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
            self.test_dataloader.collate_fn = iclevr_retrieval_collate_fn
        else:
            raise ValueError

    def generate(self, save_path, split="train"):
        # keys = [(access_id, turn_index), ...]
        if split == "train":
            keys = self.dataset.keys
            dataloader = self.dataloader
        elif split == "valid":
            keys = self.valid_dataset.keys
            dataloader = self.valid_dataloader
        elif split == "test":
            keys = self.test_dataset.keys
            dataloader = self.test_dataloader
        else:
            raise ValueError

        # all_text_features: List of Tensor (D,)
        # all_text_memories: List of Tensor (ml, D)
        # all_text_lengths: List of int [text_length, ...]
        # mb: mini-batch, ml: max dialog length in mini-batch
        all_text_features = []
        all_text_memories = []
        all_text_lengths = []

        if self.sentence_encoder_type == "rnn":
            models = (self.rnn_prev, self.rnn_curr)
        else:
            models = (self.tokenizer, self.bert)

        run_dataloader(models, self.sentence_encoder_type, dataloader, all_text_features,
                       all_text_memories, all_text_lengths)

        import h5py

        # mapping dataset_id(did) to tuple of (start_index, end_index)
        # keys = [(access_id, turn_index), ...]
        id2idxtup = defaultdict(list)
        for i, (did, tid) in enumerate(keys):
            if did not in id2idxtup:
                id2idxtup[did] = [i, i]
            else:
                id2idxtup[did][1] = i

        # create h5 datasets
        h5 = h5py.File(save_path, "w")
        for did in id2idxtup.keys():
            start, end = id2idxtup[did]
            end += 1
            # start, end = 0, 1

            # turns_text_embedding: shape=(l, D)
            # turns_text_length: shape=(l,)
            # l: dialog length
            try:
                turns_text_embedding = np.stack(
                    all_text_features[start:end], axis=0)
                turns_text_length = np.array(all_text_lengths[start:end])
            except ValueError:
                continue

            # turns_word_embeddings: shape=(l, ms, D)
            # ms: max text length of a dialog
            # it means that turns_word_embeddings is already padded.
            turns_word_embeddings = np.zeros(
                (len(turns_text_length), max(turns_text_length), self.cfg.text_dim))
            for i, j in enumerate(range(start, end)):
                try:
                    text_length = turns_text_length[i]
                    turns_word_embeddings[i, :text_length] = \
                        all_text_memories[j][:text_length]
                except:
                    pass

            scene = h5.create_group(did)
            scene.create_dataset(
                "turns_text_embedding", data=turns_text_embedding)
            scene.create_dataset(
                "turns_word_embeddings", data=turns_word_embeddings)
            scene.create_dataset(
                "turns_text_length", data=turns_text_length)


def create_embs_from_model(cfg):
    logger.info(f"script {__name__} start!")

    generator = SentenceEmbeddingGenerator(cfg)

    for split in ["train", "valid", "test"]:
        save_path = os.path.join(
            cfg.save_root_dir,
            f"{cfg.dataset}_{split}_embeddings_{cfg.fork}.h5",
        )
        logger.info(f"create additional dataset: {save_path}")
        generator.generate(save_path, split)
