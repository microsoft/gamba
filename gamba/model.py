from typing import Dict, Optional, Set, Tuple, Type
import math
import numpy as np
from sequence_models.constants import MSA_PAD, START, STOP
from sequence_models.convolutional import ByteNetBlock, ByteNetLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from torch.distributions import Gamma, Normal


from gamba.constants import MSA_ALPHABET_PLUS, TaskType
from gamba.losses import (
    OAMaskedCrossEntropyLoss,
    GaussianNLLLoss,
    InverseGammaNLLLoss,
    PoissonNLLLoss,
)


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
        self.embedder = ARDiffusionModel(jambalm).module.model

        # need to split d_model into lm head, scaling head and gap head
        # self.each_dim = int(d_model / 3)
        self.each_dim = int(d_model / 2)
        self.lm_head = nn.Linear(d_model, jambalm.vocab_size)
        self.scaling_head = nn.Linear(d_model, 2)
        # self.gap_head = nn.Linear(self.each_dim, 1)

        # self.down = nn.Linear(
        #     2 * jambalm.model.embed_tokens.weight.shape[-1] + d_model, d_model
        # )

        #seq_embedding gets full dimensionality, there is no more any value embedding
        self.seq_embedding = nn.Embedding(jambalm.vocab_size, d_model)
        #self.value_embedding = nn.Linear(1, self.each_dim)
        

        # real number loss
        self.cons_loss_func = GaussianNLLLoss()
        # self.gap_loss_func = PoissonNLLLoss()
        # # self.embed_tokens = nn.Embedding(jambalm.vocab_size, d_model)
        # self.padding_id = padding_id
        # self.decoder = nn.Linear(2 * jambalm.model.embed_tokens.weight.shape[-1], jambalm.vocab_size)
        # self.decoder = nn.Linear(2 * jambalm.vocab_size, jambalm.vocab_size)
        # self.lm_head = self.jambalm.module.lm_head

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        # print("shape of tgt: ", tgt.shape)
        # print("shape of src: ", src.shape)
        # seq_tgt, conservation_tgt, gap_tgt = tgt.split(1, dim=0)
        seq_tgt, conservation_tgt = tgt.split(1, dim=1)
        seq_tgt = seq_tgt.squeeze(1).long()
        conservation_tgt = conservation_tgt.squeeze(1)
        #print(f"conservation tgt shape:", conservation_tgt.shape)
        #print(f"seq tgt shape:", seq_tgt.shape)
        # gap_tgt = gap_tgt.squeeze(0)
        # 0 is a token not a padding
        n_tokens = (seq_tgt >= 0).sum()

        # seq, conservation, gap = src.split(1, dim=0)
        seq, conservation = src.split(1, dim=1)
        #print("shape of seq: ", seq.shape)
        seq = seq.squeeze(1).long()
        #print("post squeeze & long shape of seq: ", seq.shape)
        conservation = conservation.squeeze(1)
        #print(f"conservation shape:", conservation.shape)
        # gap = gap.squeeze(0)
        device = src.device

        n_seq = torch.tensor(seq.size(1), device=device)
        n_processed = n_tokens - seq_tgt.size(1)  # -1 token per sequence for the shift

        # embed seq, conservation and gap separately
        emb_seq = self.seq_embedding(seq)
        #print(f"embed seq/input embeds shape:", emb_seq.shape)

        # gap has shape (batch, seq_length)
        # gap_reshaped = gap.view(-1, 1)  # reshape to (seq_length, batch)
        # embed
        # emb_gap = self.value_embedding(gap_reshaped)
        # reshape the output back to (batch, seq_length, embedding dim)
        # emb_gap = emb_gap.view(1, -1, self.each_dim)


        #no more conservation embedding
        # conservation_reshaped = conservation.reshape(
        #     -1, 1
        # )  # reshape to (batch * seq_length, 1)
        # emb_conservation = self.value_embedding(conservation_reshaped)
        # emb_conservation = emb_conservation.view(
        #     seq.size(0), -1, self.each_dim
        # )  # reshape back to (batch, seq_length, embedding_dim)

        # print(
        #     f"shapes of emb_seq, emb_conservation, emb_gap: {emb_seq.shape}, {emb_conservation.shape}, {emb_gap.shape}"
        # )
        #next, concatenate the embeddings along the hidden dimension and send to the model
        #inputs_embeds = torch.cat([emb_seq, emb_conservation, emb_gap], dim=-1)

        #no more conservation embedding
        #inputs_embeds = torch.cat([emb_seq, emb_conservation], dim=-1)

        inputs_embeds = emb_seq
        #inputs_embeds= torch.transpose(emb_seq, 1, 2)

        #print(f"shape of input_embeds: {inputs_embeds.shape}")
        # need to set the embedded inputs to inputs_embeds to values in the Jamba model
        output = self.embedder(inputs_embeds=inputs_embeds)["last_hidden_state"]
        # take the output of the model and split it along the last dimension
        #seq_output, scaling_output = output.split(output.shape[-1] // 2, dim=-1)
        # seq_output, scaling_output, gap_output = output.split(
        #     output.shape[-1] // 3, dim=-1
        # )
        # put the outputs through their respective linear layers
        seq_logits = self.lm_head(output)
        scaling_logits = self.scaling_head(output)
        # gap_logits = self.gap_head(gap_output)
        # print(
        #     f"shapes of seq_logits, scaling_logits, gap_logits: {seq_logits.shape}, {scaling_logits.shape},"  # {gap_logits.shape}"
        # )
        # print(
        #     f"seq_logits: {seq_logits}, scaling_logits: {scaling_logits}"  # , gap_logits: {gap_logits}"
        # )

        # exclude the logits for the first and last tokens for conservation and gap
        scaling_logits = scaling_logits[:, 1:-1]
        # gap_logits = gap_logits[:, 1:-1]
        conservation_tgt = conservation_tgt[:, 1:-1]
        # gap_tgt = gap_tgt[:, 1:-1]

        # apply CE loss on the seq_logits
        ce_loss = F.cross_entropy(
            seq_logits[:, :-1, :].reshape(-1, seq_logits.shape[-1]),
            seq_tgt[:, 1:].flatten(),
            reduction="mean",
        )
        # apply GaussianNLLLoss from losses.py on the scaling_logits
        gaussian_loss = self.cons_loss_func(
            scaling_logits[:, :-1, :], conservation_tgt[:, 1:]
        )
        #clip the gaussian loss
        #gaussian_loss = torch.clamp(gaussian_loss, min=0.0, max=1.0)
        # apply PoissonNLLLoss from losses.py on the gap_logits
        # poisson_loss = self.gap_loss_func(gap_logits[:, :-1, :], gap_tgt[:, 1:])

        #print("CE LOSS: ", ce_loss)
        #print("GAUSSIAN LOSS: ", gaussian_loss)
        # check if any loss is NaN
        if math.isnan(ce_loss):
            raise ValueError("CE Loss is NaN")
        if math.isnan(gaussian_loss):
            raise ValueError("Gaussian Loss is NaN")
        # print("POISSON LOSS: ", poisson_loss)
        #print("LOSS:", ce_loss + gaussian_loss)

        # print(f"shape of the gap_logits: {gap_logits.shape}")
        # print(f"shape of the scaling_logits: {scaling_logits.shape}")
        # print(f"shape of the seq_logits: {seq_logits.shape}")
        # compute the accuracy
        with torch.no_grad():
            pred_tok_seq = torch.argmax(seq_logits[:, :-1, :], dim=-1)
            seq_accu = (
                (pred_tok_seq == seq_tgt[:, 1:]) * (seq_tgt[:, 1:] >= 0)
            ).float().sum() / n_tokens

        other_metrics = {
            "accuracy": seq_accu,
        }
        if hasattr(output, "aux_loss"):
            # log the original CE loss and the auxiliary loss
            other_metrics["ce_loss"] = ce_loss
        outputs = {
            "seq_logits": seq_logits,
            "scaling_logits": scaling_logits,
            # "gap_logits": gap_logits,
            "loss": ce_loss + gaussian_loss,  # + poisson_loss,
            "cross_entropy_loss": ce_loss,
            "gaussian_loss": gaussian_loss,
            OTHER_METRICS_KEY: other_metrics,
            "accuracy": seq_accu,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
            "representation": output,
            "conservation_tgt": conservation_tgt,
        }
        return outputs

class JambaGambaModelWithDegeneracies(nn.Module):
    def __init__(
        self,
        jambalm: nn.Module,
        d_model: int,
        nhead: int,
        dim_feedfoward: int,
        n_layers: int,
        padding_id: int,
    ):
        super(JambaGambaModelWithDegeneracies, self).__init__()
        self.embedder = ARDiffusionModel(jambalm).module.model

        # need to split d_model into lm head, scaling head
        self.each_dim = int(d_model / 2)
        self.lm_head = nn.Linear(d_model, jambalm.vocab_size)
        self.scaling_head = nn.Linear(d_model, 2)
       

        #seq_embedding gets full dimensionality, there is no more any value embedding
        self.seq_embedding = nn.Embedding(jambalm.vocab_size, d_model)
        

        # real number loss
        self.cons_loss_func = GaussianNLLLoss()

     

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> dict:
        # Split the target tensor into sequence and conservation targets
        seq_tgt, conservation_tgt, degeneracies_tgt = tgt.split(1, dim=1)
        seq_tgt = seq_tgt.squeeze(1).long()
        conservation_tgt = conservation_tgt.squeeze(1)
        degeneracies_tgt = degeneracies_tgt.squeeze(1)
        n_tokens = (seq_tgt >= 0).sum()

        # Split the source tensor into sequence and conservation inputs
        seq, conservation, degeneracies = src.split(1, dim=1)
        seq = seq.squeeze(1).long()
        conservation = conservation.squeeze(1)
        degeneracies = degeneracies.squeeze(1)
        device = src.device

        n_seq = torch.tensor(seq.size(1), device=device)
        n_processed = n_tokens - seq_tgt.size(1)  # -1 token per sequence for the shift

        # Embed the sequence
        emb_seq = self.seq_embedding(seq)
        inputs_embeds = emb_seq

        # Pass the embedded inputs through the model
        output = self.embedder(inputs_embeds=inputs_embeds)["last_hidden_state"]

        # Generate logits for sequence and scaling
        seq_logits = self.lm_head(output)
        scaling_logits = self.scaling_head(output)

        # Exclude the logits for the first and last tokens for conservation
        scaling_logits = scaling_logits[:, 1:-1]
        conservation_tgt = conservation_tgt[:, 1:-1]
        degeneracies_tgt = degeneracies_tgt[:, 1:-1]

        # Apply cross-entropy loss on the sequence logits
        ce_loss = F.cross_entropy(
            seq_logits[:, :-1, :].reshape(-1, seq_logits.shape[-1]),
            seq_tgt[:, 1:].flatten(),
            reduction="mean",
        )

        # Apply GaussianNLLLoss on the scaling logits
        gaussian_loss = self.cons_loss_func(
            scaling_logits[:, :-1, :], conservation_tgt[:, 1:]
        )

        # Check if any loss is NaN
        if math.isnan(ce_loss):
            raise ValueError("CE Loss is NaN")
        if math.isnan(gaussian_loss):
            raise ValueError("Gaussian Loss is NaN")

        # Compute the accuracy
        with torch.no_grad():
            pred_tok_seq = torch.argmax(seq_logits[:, :-1, :], dim=-1)
            seq_accu = (
                (pred_tok_seq == seq_tgt[:, 1:]) * (seq_tgt[:, 1:] >= 0)
            ).float().sum() / n_tokens

        other_metrics = {
            "accuracy": seq_accu,
        }
        if hasattr(output, "aux_loss"):
            other_metrics["ce_loss"] = ce_loss

        outputs = {
            "seq_logits": seq_logits,
            "scaling_logits": scaling_logits,
            "loss": ce_loss + gaussian_loss,
            "cross_entropy_loss": ce_loss,
            "gaussian_loss": gaussian_loss,
            OTHER_METRICS_KEY: other_metrics,
            "accuracy": seq_accu,
            "n_tokens": n_tokens,
            "n_seqs": n_seq,
            "n_processed": n_processed,
            "representation": output,
            "conservation_tgt": conservation_tgt,
            "degeneracies_tgt": degeneracies_tgt,
        }
        return outputs
    
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
