import numpy as np

import torch
from tqdm import tqdm
from tpu_device import tpu_device

import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


# def map_device_loader(idx, sentence_encoder_type, dataloader, models):
#     # device = xm.xla_device(n=idx)
#     loader = dataloader
#     # loader = pl.MpDeviceLoader(dataloader, device)

#     results = []

#     for batch in tqdm(loader):
#         if sentence_encoder_type == "rnn":
#             rnn_prev, rnn_curr = models
#             # extract from batch
#             prev_embs = batch["prev_embs"]
#             embs = batch["embs"]
#             prev_seq_len = batch["prev_seq_len"]
#             seq_len = batch["seq_len"]

#             # forward sentence encoder
#             _, _, context = rnn_prev(
#                 prev_embs, prev_seq_len)
#             text_memories, text_feature, key_padding_mask, seq_len = rnn_curr(
#                 embs, seq_len, context)
#         elif sentence_encoder_type == "bert":
#             tokenizer, bert = models
#             prev_utter = batch["prev_utter"]
#             utter = batch["utter"]

#             inputs = tokenizer(
#                 text=prev_utter,
#                 text_pair=utter,
#                 add_special_tokens=True,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#             )

#             text_memories, text_feature, key_padding_mask = bert(
#                 inputs["input_ids"].to(tpu_device),
#                 inputs["attention_mask"].to(tpu_device),
#                 inputs["token_type_ids"].to(tpu_device),
#             )
#             seq_len = None

#         results.append((text_memories, text_feature,
#                        key_padding_mask, seq_len))

#     return results


# def run_dataloader(models, sentence_encoder_type, dataloader, all_text_features, all_text_memories, all_text_lengths):
#     with torch.no_grad():
#         results = map_device_loader(
#             0, sentence_encoder_type, dataloader, models)
#         # results = xmp.spawn(map_device_loader,
#         #                     args=(sentence_encoder_type, dataloader, models), nprocs=1, start_method='fork')

#         for text_memories, text_feature, key_padding_mask, seq_len in results:
#             # push to list
#             for i in range(text_memories.size(0)):
#                 if sentence_encoder_type == "rnn":
#                     all_text_features.append(
#                         text_feature[i].cpu().numpy())
#                     all_text_memories.append(
#                         text_memories[i].cpu().numpy())
#                     all_text_lengths.append(
#                         seq_len[i].cpu().numpy())
#                 elif sentence_encoder_type == "bert":
#                     all_text_features.append(
#                         text_feature[i])

#                     # output text memories of bert includes embedding of previous instructions.
#                     # key_padding_mask is False where [CLS] & [[SENTENCE B], ...] & [SEP]
#                     bool_indices = ~key_padding_mask[i]
#                     _seq_len = np.array(bool_indices.sum().item())
#                     _text_memories = text_memories[i][bool_indices]
#                     _text_memories = text_memories[i][bool_indices]

#                     all_text_memories.append(
#                         _text_memories)
#                     all_text_lengths.append(
#                         _seq_len)

def run_dataloader(models, sentence_encoder_type, dataloader, all_text_features, all_text_memories, all_text_lengths):
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if sentence_encoder_type == "rnn":
                rnn_prev, rnn_curr = models
                # extract from batch
                prev_embs = batch["prev_embs"]
                embs = batch["embs"]
                prev_seq_len = batch["prev_seq_len"]
                seq_len = batch["seq_len"]

                # forward sentence encoder
                _, _, context = rnn_prev(
                    prev_embs, prev_seq_len)
                text_memories, text_feature, key_padding_mask, seq_len = rnn_curr(
                    embs, seq_len, context)
            elif sentence_encoder_type == "bert":
                tokenizer, bert = models
                prev_utter = batch["prev_utter"]
                utter = batch["utter"]

                inputs = tokenizer(
                    text=prev_utter,
                    text_pair=utter,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                text_memories, text_feature, key_padding_mask = bert(
                    inputs["input_ids"].to(tpu_device),
                    inputs["attention_mask"].to(tpu_device),
                    inputs["token_type_ids"].to(tpu_device),
                )
                seq_len = None

            # push to list
            for i in range(text_memories.size(0)):
                if sentence_encoder_type == "rnn":
                    all_text_features.append(
                        text_feature[i].cpu().numpy())
                    all_text_memories.append(
                        text_memories[i].cpu().numpy())
                    all_text_lengths.append(
                        seq_len[i].cpu().numpy())
                elif sentence_encoder_type == "bert":
                    all_text_features.append(
                        text_feature[i].cpu().numpy())

                    # output text memories of bert includes embedding of previous instructions.
                    # key_padding_mask is False where [CLS] & [[SENTENCE B], ...] & [SEP]
                    bool_indices = ~key_padding_mask[i]
                    _seq_len = np.array(bool_indices.sum().item())
                    _text_memories = text_memories[i][bool_indices]

                    all_text_memories.append(
                        _text_memories.cpu().numpy())
                    all_text_lengths.append(
                        _seq_len)
