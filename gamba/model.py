from typing import Dict, Optional, Set, Tuple, Type
import math
import numpy as np
from sequence_models.constants import MSA_PAD, START, STOP
from sequence_models.convolutional import ByteNetBlock, ByteNetLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from gamba.constants import MSA_ALPHABET_PLUS, TaskType
from gamba.losses import OAMaskedCrossEntropyLoss


OTHER_METRICS_KEY = "other_metrics"


class LogitOnlyModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return {"logits": self.module(*args, **kwargs)}


class OrderAgnosticDiffusionModel(nn.Module):
    def __init__(
        self, module: nn.Module, padding_id: int, aux_loss_weight: float = 1.0
    ):
        super().__init__()
        self.module = module
        self.loss_func = OAMaskedCrossEntropyLoss(reweight=True)
        self.padding_id = padding_id
        self.aux_loss_weight = aux_loss_weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        n_tokens = mask.sum()
        input_mask = src != self.padding_id
        n_seq = torch.tensor(len(src), device=src.device)
        n_processed = input_mask.sum()

        output = self.module(src, input_mask=input_mask.unsqueeze(-1))
        ce_loss, nll_loss = self.loss_func(
            output["logits"], tgt, mask, timestep, input_mask
        )
        aux_loss = output.get("aux_loss", 0.0)

        with torch.no_grad():
            pred_tok = torch.argmax(output["logits"], dim=-1)
            accu = ((pred_tok == tgt) * mask).float().sum() / n_tokens

        ce_loss = ce_loss / n_tokens
        nll_loss = nll_loss / n_tokens
        other_metrics = {
            "nll_loss": nll_loss,
            "accuracy": accu,
        }
        if hasattr(output, "aux_loss"):
            # log the original CE loss and the auxiliary loss
            other_metrics["ce_loss"] = ce_loss
            other_metrics["aux_loss"] = aux_loss

        outputs = {
            "logits": output["logits"],
            "loss": ce_loss + self.aux_loss_weight * aux_loss,
            OTHER_METRICS_KEY: other_metrics,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
        }
        return outputs


class ARDiffusionModel(nn.Module):
    def __init__(self, module: nn.Module, aux_loss_weight: float = 1.0):
        super().__init__()
        self.module = module
        self.aux_loss_weight = aux_loss_weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        n_tokens = (tgt >= 0).sum()
        n_seq = torch.tensor(len(src), device=src.device)
        n_processed = n_tokens - len(tgt)  # -1 token per sequence for the shift

        output = self.module(src)
        ce_loss = F.cross_entropy(
            # flatten into N*L x C
            output["logits"][:, :-1, :].reshape(-1, output["logits"].shape[-1]),
            tgt[:, 1:].flatten(),
            reduction="mean",
        )
        aux_loss = output.get("aux_loss", 0.0)

        # compute the accuracy
        with torch.no_grad():
            pred_tok = torch.argmax(output["logits"][:, :-1, :], dim=-1)
            accu = (
                (pred_tok == tgt[:, 1:]) * (tgt[:, 1:] >= 0)
            ).float().sum() / n_tokens

        other_metrics = {
            "accuracy": accu,
        }
        if hasattr(output, "aux_loss"):
            # log the original CE loss and the auxiliary loss
            other_metrics["ce_loss"] = ce_loss
            other_metrics["aux_loss"] = aux_loss
        outputs = {
            "logits": output["logits"],
            "loss": ce_loss + self.aux_loss_weight * aux_loss,
            OTHER_METRICS_KEY: other_metrics,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
        }
        return outputs


def get_flip_inds(seq, lengths):
    # hidden_states: b x ell
    # lengths: b x 1
    device = seq.device
    b, ell = seq.shape[:2]
    xi = torch.arange(b, device=device).repeat_interleave(ell, dim=0).view(b, ell)
    yi = (torch.arange(ell, device=device) + (ell - lengths)) % ell
    return xi, yi


def flip_with_padding(seq, lengths):
    xi, yi = get_flip_inds(seq, lengths)
    qes = seq.flip(dims=(1,))
    return qes[xi, yi]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0, : x.size(1)]
        return self.dropout(x)


class JambagambaModel(nn.Module):
    def __init__(
        self,
        jambalm: nn.Module,
        d_model: int,
        nhead: int,
        dim_feedfoward: int,
        n_layers: int,
        padding_id: int,
    ):
        super().__init__()
        self.jambalm = ARDiffusionModel(jambalm)
        self.embedder = self.jambalm.module.model
        # need to split d_model into lm head, scaling head and error head
        self.lm_head = nn.Linear(d_model, jambalm.vocab_size)
        self.scaling_head = nn.Linear(d_model, 1)
        self.error_head = nn.Linear(d_model, 1)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedfoward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, n_layers)
        self.down = nn.Linear(
            2 * jambalm.model.embed_tokens.weight.shape[-1] + d_model, d_model
        )
        self.pe = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(jambalm.vocab_size, d_model)
        # # self.embed_tokens = nn.Embedding(jambalm.vocab_size, d_model)
        # self.padding_id = padding_id
        # self.decoder = nn.Linear(2 * jambalm.model.embed_tokens.weight.shape[-1], jambalm.vocab_size)
        # self.decoder = nn.Linear(2 * jambalm.vocab_size, jambalm.vocab_size)
        # self.lm_head = self.jambalm.module.lm_head

    def forward(
        self, src: torch.Tensor, input_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        ells = input_mask.sum(dim=1)  # b x 1
        b, input_length = src.shape
        print(
            f"IN FORWARD JAMBAGAMBA, HAVE BATCH B: {b} AND INPUT_LENGTH: {input_length}, and SRC.SHAPE: {src.shape}"
        )
        device = src.device
        print(f"DEVICE: {device}")
        print(f"ells: {ells}, {len(ells)}")
        print(f"ells.squeeze(): {ells.squeeze()}")
        with torch.no_grad():
            keep_x = (
                torch.arange(b, device=device)
                .repeat_interleave(input_length - 2, dim=0)
                .view(b, input_length - 2)
            )
            y1 = [torch.arange(ell - 2, device=device) for ell in ells.squeeze()]
            y2 = [
                torch.arange(ell, input_length, device=device) for ell in ells.squeeze()
            ]
            keep_y = torch.stack(
                [torch.cat([y1i, y2i], dim=-1) for y1i, y2i in zip(y1, y2)]
            )
            e_fwd = self.embedder(src)["last_hidden_state"]
            e_fwd = e_fwd[keep_x, keep_y]
            crs = flip_with_padding(src, ells)
            e_rev = self.embedder(crs)["last_hidden_state"]
            e_rev = e_rev[keep_x, keep_y]
            e_rev = flip_with_padding(e_rev, ells - 2)
            e = torch.cat([e_fwd, e_rev], dim=-1)
        s = self.embedding(src)
        s = s[keep_x, keep_y]
        e = torch.cat([e, s], dim=-1)
        e = self.down(e)
        e = self.pe(e)
        e = self.decoder(e, src_key_padding_mask=~input_mask.squeeze(-1)[:, 2:])
        logits = self.lm_head(e)
        return logits


def _create_bytenet(
    task: TaskType, model_config: dict, pad_id: int
) -> Tuple[ByteNetLM, Set[Type[ByteNetBlock]]]:
    pretrained = model_config.pop("pretrained", False)
    if pretrained:
        raise ValueError("Pretrained models not supported for ByteNet")

    n_tokens = len(MSA_ALPHABET_PLUS)
    d_embed = model_config["d_embed"]
    d_model = model_config["d_model"]
    n_layers = model_config["n_layers"]
    kernel_size = model_config["kernel_size"]
    r = model_config["r"]
    slim = model_config.get("slim", True)
    activation = model_config.get("activation", "gelu")
    dropout = model_config.get("dropout", 0.0)

    return (
        ByteNetLM(
            n_tokens,
            d_embed,
            d_model,
            n_layers,
            kernel_size,
            r,
            causal=task == TaskType.LM,
            padding_idx=pad_id,
            dropout=dropout,
            tie_weights=False,
            final_ln=True,
            slim=slim,
            activation=activation,
        ),
        {ByteNetBlock},
    )


def _get_hf_model(
    model_name: str,
    pad_token_id: int,
    *,
    model_config: Optional[dict] = None,
    pretrained: bool = False,
    trust_remote_code: bool = False,
) -> nn.Module:
    if model_config and pretrained:
        # can't overwrite the config of a pretrained model
        raise ValueError("Cannot specify both model_config and pretrained")
    elif pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # if we need to change the padding token
        if pad_token_id != model.config.pad_token_id:
            # locate the single embedding module
            embeddings = []
            for layer in model.modules():
                if isinstance(layer, nn.Embedding):
                    embeddings.append(layer)
            if len(embeddings) != 1:
                raise ValueError(f"Expected 1 embedding layer, got {len(embeddings)}")

            # update the padding index
            embeddings[0].padding_idx = pad_token_id
            embeddings[0]._fill_padding_idx_with_zero()
    else:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        for k, v in model_config.items():
            if not hasattr(config, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(config, k, v)

        # ensure the vocab size is a multiple of 8 to maximize tensor core utilization
        model_config["vocab_size"] = (
            np.ceil(len(MSA_ALPHABET_PLUS) / 8).astype(int).item() * 8
        )
        model_config["pad_token_id"] = MSA_ALPHABET_PLUS.index(
            MSA_PAD
        )  # FIXME: MSA_PAD or pad_token_id (which is mask_id in bytenet
        model_config["bos_token_id"] = MSA_ALPHABET_PLUS.index(START)
        model_config["eos_token_id"] = MSA_ALPHABET_PLUS.index(STOP)

        # merge the updates into the default config
        config = type(config).from_dict({**config.to_dict(), **model_config})
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code
        )
    return model


def _create_jamba(
    task: TaskType, model_config: dict, pad_id: int
) -> Tuple[nn.Module, Set[Type[nn.Module]]]:
    pretrained = model_config.pop("pretrained", False)
    model = _get_hf_model(
        "ai21labs/Jamba-v0.1", pad_id, pretrained=pretrained, model_config=model_config
    )
    return model, {type(layer) for layer in model.model.layers}


def create_model(
    task: TaskType, model_type: str, model_config: dict, pad_id: int
) -> Tuple[nn.Module, Set[Type[nn.Module]]]:
    if model_type == "bytenet":
        model, blocks = _create_bytenet(task, model_config, pad_id)
    elif model_type == "jamba":
        model, blocks = _create_jamba(task, model_config, pad_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # assume all non-HF models only output logits, so we wrap it
    # to make it look like a HF model
    if not isinstance(model, PreTrainedModel):
        model = LogitOnlyModelWrapper(model)

    return model, blocks
