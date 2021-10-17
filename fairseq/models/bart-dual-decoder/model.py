"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension

BART-dual-decoder is a dual-decoder version of BART.
This model is used for keyphrase research with multitask objectives.
When the auxiliary task is disabled, the model is equivalent to BART.
"""

import logging
import copy

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import BARTDualDecoderHubInterface


logger = logging.getLogger(__name__)


@register_model("bart-dual-decoder")
class BARTDualDecoderModel(TransformerModel):
    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz"
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

        assert args.dual_decoder_scheme

    @staticmethod
    def add_args(parser):
        super(BARTDualDecoderModel, BARTDualDecoderModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )
        # arguments related to dual decoder
        parser.add_argument(
            "--dual-decoder-scheme",
            action='store_true',
            default=False,
            help="Set False to disable decoder 1.",
        )
        parser.add_argument(
            "--disable-decoder1",
            action='store_true',
            default=False,
            help="Set False to disable decoder 1.",
        )
        parser.add_argument(
            "--disable-decoder2",
            action='store_true',
            default=False,
            help="Set False to disable decoder 2.",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens1,
        prev_output_tokens2,
        features_only=False,
        classification_head_name=None,
        token_embeddings=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            **kwargs,
        )
        x1, x2, extra1, extra2 = self.decoder(
            prev_output_tokens1,
            prev_output_tokens2,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        # if classification_head_name is not None:
        #     sentence_representation = x[
        #         src_tokens.eq(self.encoder.dictionary.eos()), :
        #     ].view(x.size(0), -1, x.size(-1))[:, -1, :]
        #     x = self.classification_heads[classification_head_name](
        #         sentence_representation
        #     )
        return x1, x2, extra1, extra2

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])

    # def register_classification_head(
    #     self, name, num_classes=None, inner_dim=None, **kwargs
    # ):
    #     """Register a classification head."""
    #     logger.info("Registering classification head: {0}".format(name))
    #     if name in self.classification_heads:
    #         prev_num_classes = self.classification_heads[name].out_proj.out_features
    #         prev_inner_dim = self.classification_heads[name].dense.out_features
    #         if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
    #             logger.warning(
    #                 're-registering head "{}" with num_classes {} (prev: {}) '
    #                 "and inner_dim {} (prev: {})".format(
    #                     name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
    #                 )
    #             )
    #     self.classification_heads[name] = BARTClassificationHead(
    #         input_dim=self.args.encoder_embed_dim,
    #         inner_dim=inner_dim or self.args.encoder_embed_dim,
    #         num_classes=num_classes,
    #         activation_fn=self.args.pooler_activation_fn,
    #         pooler_dropout=self.args.pooler_dropout,
    #         do_spectral_norm=self.args.spectral_norm_classification_head,
    #     )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
            self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                -1, :
            ]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
        

        # for dual decoders, we check whether we are actually loading from a single decoder
        # if so, we load the same weight to both the decoders (with deep copy)
        original_keys = list(state_dict.keys())
        if not any(x for x in original_keys if ('decoder1' in x or 'decoder2' in x) and 'weight' in x):
            for x in original_keys:
                if 'decoder' in x:
                    if 'version' in x:
                        state_dict[x] = state_dict['encoder.version']
                    else:
                        weight_val = state_dict.pop(x)
                        new_name_1 = x.replace('decoder', 'decoder.decoder1')
                        new_name_2 = x.replace('decoder', 'decoder.decoder2')
                        state_dict[new_name_1] = weight_val
                        state_dict[new_name_2] = copy.deepcopy(state_dict[new_name_1])
            
            if 'decoder.decoder1.output_projection.weight' not in state_dict:
                state_dict['decoder.decoder1.output_projection.weight'] = state_dict['encoder.embed_tokens.weight']
                state_dict['decoder.decoder2.output_projection.weight'] = state_dict['encoder.embed_tokens.weight']
            
            state_dict.pop('decoder.version')

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        # if hasattr(self, "classification_heads"):
        #     cur_state = self.classification_heads.state_dict()
        #     for k, v in cur_state.items():
        #         if prefix + "classification_heads." + k not in state_dict:
        #             logger.info("Overwriting", prefix + "classification_heads." + k)
        #             state_dict[prefix + "classification_heads." + k] = v


# class BARTClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""

#     def __init__(
#         self,
#         input_dim,
#         inner_dim,
#         num_classes,
#         activation_fn,
#         pooler_dropout,
#         do_spectral_norm=False,
#     ):
#         super().__init__()
#         self.dense = nn.Linear(input_dim, inner_dim)
#         self.activation_fn = utils.get_activation_fn(activation_fn)
#         self.dropout = nn.Dropout(p=pooler_dropout)
#         self.out_proj = nn.Linear(inner_dim, num_classes)

#         if do_spectral_norm:
#             self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

#     def forward(self, features, **kwargs):
#         x = features
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = self.activation_fn(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


@register_model_architecture("bart-dual-decoder", "bart-dual-decoder-large")
def bart_dual_decoder_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)


@register_model_architecture("bart-dual-decoder", "bart-dual-decoder-base")
def bart_dual_decoder_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_dual_decoder_large_architecture(args)
