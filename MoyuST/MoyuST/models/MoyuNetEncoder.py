import torch
import torch.nn as nn
import numpy as np
import os
from fairseq.models.speech_to_text.xstnet import XSTNetEncoder
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.hubert import HubertModel
import logging

logger = logging.getLogger(__name__)

class MoyuAttnLayer(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.input_size = input_size
        self.expand_size = input_size * 2
        self.expand_adapter = self._build_ffn(self.input_size, self.expand_size)
        self.forget_gate = self._build_gate(self.expand_size)
        self.update_gate = self._build_gate(self.expand_size)
        self.multi_head_attn = MultiheadAttention(
            self.expand_size,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        self.shrink_adapter = self._build_ffn(self.expand_size, self.input_size)
    
    def _build_ffn(self, input_size, expand_size):
        ffn_layer = nn.Linear(input_size, expand_size)
        relu = nn.ReLU()
        nn.init.xavier_uniform_(ffn_layer.weight)
        return nn.Sequential(ffn_layer, relu)
    
    def _build_gate(self, input_size):
        ffn_layer = nn.Linear(input_size, 1)
        sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(ffn_layer.weight)
        return nn.Sequential(ffn_layer, sigmoid)
    
    def forward(self, x, updates=None):
        x = self.expand_adapter(x)
        forget_value = self.forget_gate(x)
        update_value = self.update_gate(x)
        attn_x, _ = self.multi_head_attn(x, x, x)
        gated_value = attn_x + update_value * attn_x + (1 - forget_value) * attn_x
        # if updates % 100 == 0:
        #     logger.info(f"forget: max: {torch.max(forget_value)}, min: {torch.min(forget_value)}, mean: {torch.mean(forget_value)}")
        #     logger.info(f"update: max: {torch.max(update_value)}, min: {torch.min(update_value)}, mean: {torch.mean(update_value)}")
        #     logger.info(f"gated: max: {torch.max(gated_value)}, min: {torch.min(gated_value)}, mean: {torch.mean(gated_value)}")
        x = self.shrink_adapter(gated_value)
        return x

class MoyuNetEncoder(XSTNetEncoder):
    def __init__(self, args, dict, embed_tokens):
        super().__init__(args, dict, embed_tokens)
        self.using_moyulayer = args.using_attn
        if self.using_moyulayer:
            self.Moyu_layer = MoyuAttnLayer(args, self.args.encoder_embed_dim)
    
    def _build_acoustic_encoder(self, args):
        assert args.hubert_model_path is not None
        self.hubert_model_path = args.hubert_model_path
        self.use_asr_finetune_hubert = args.use_asr_finetune_hubert
        try:
            ckpt = torch.load(self.hubert_model_path)
        except FileNotFoundError:
            if not os.path.exists("hubert_base_ls960.pt"):
                assert 1==2, "Please download hubert_base_ls960.pt from https://dl.fbaipublicfiles.com/fairseq/hubert/hubert_base_ls960.pt and put it in the current directory"
            ckpt = torch.load("hubert_base_ls960.pt")
        self.hubert_args = ckpt["args"]
        if not self.use_asr_finetune_hubert:  # if use ssl-trained only
            self.hubert_args = ckpt["args"]
            self.hubert_model = HubertModel.build_model(ckpt['args'], dictionary=[torch.rand(504)])
            self.hubert_model.load_state_dict(ckpt['model'])
        else:  # hubert-ctc model
            assert 1==2, "not implemented yet"
        self.freeze_hubert = args.freeze_hubert

        hubert_output_dim = self.hubert_args.encoder_embed_dim
        self.subsample_audio = Conv1dSubsampler(
            hubert_output_dim,
            args.conv_channels,
            self.textual_encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )
    
    def _get_hubert_feature(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        hubert_feature, padding_mask = self.hubert_model.extract_features(src_tokens, padding_mask)
        output_length = (1 - padding_mask.int()).sum(dim=1)
        return hubert_feature, padding_mask, output_length

    def embedding_audio(self, src_tokens, src_lengths,
                        return_short_audio_len=False):
        if self.freeze_hubert:
            with torch.no_grad():
                hubert_feature, encoder_padding_mask, input_lengths = self._get_hubert_feature(
                    src_tokens, src_lengths)
        else:
            hubert_feature, encoder_padding_mask, input_lengths = self._get_hubert_feature(
                src_tokens, src_lengths)

        x, input_lengths = self.subsample_audio(hubert_feature, input_lengths)
        x = self.embed_scale * x
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if self.embed_positions is not None:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
        x = self.dropout_module(x)
        if return_short_audio_len:
            return x, encoder_padding_mask, input_lengths
        return x, encoder_padding_mask, None

    def forward(self, src_tokens, src_lengths, is_text_input=False, **kwargs):
        """
        src_tokens: b x seq, float tensor if it is audio input, LongTensor if it is text input
        src_lengths: b-dim LongTensor
        """
        short_audio_len = None
        if self.is_text_input:
            is_text_input = True
        if is_text_input:
            x, encoder_padding_mask = self.embedding_text(src_tokens, src_lengths)
        else:
            x, encoder_padding_mask, short_audio_len = self.embedding_audio(src_tokens, src_lengths,
                                                                            return_short_audio_len=True)
            if self.using_moyulayer:
                x = self.Moyu_layer(x)
        encoder_embedding = x
        # 3. Transformer-layers
        # layer_attn_weights = []
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            # layer_attn_weights.append(attn)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=encoder_embedding,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
            output_encoder_lengths=short_audio_len,
            # attn_weights=layer_attn_weights
        )