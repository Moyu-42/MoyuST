import logging

from pathlib import Path
from typing import OrderedDict
from fairseq import checkpoint_utils, utils, tasks
from fairseq.models.speech_to_text.xstnet import XSTNet, base_architecture
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from .MoyuNetEncoder import MoyuNetEncoder
from .MoyuNetDecoder import MoyuNetDecoder

logger = logging.getLogger(__name__)

@register_model('moyunet')
class MoyuNet(XSTNet):
    @staticmethod
    def add_args(parser):
        XSTNet.add_args(parser)
        parser.add_argument("--using-attn", action="store_true",
                            help="using attention as feature extract")
        parser.add_argument("--hubert-model-path", type=str, metavar="N",
                            help="path/to/hubert/model, support hdfs")
        parser.add_argument("--freeze-hubert", action="store_true",
                            help="if we want to freeze the hubert features")
        parser.add_argument("--use-asr-finetune-hubert", action="store_true",
                            help="if we want to load hubert asr finetuned data")
        parser.add_argument(
            "--load-pretrained-mt-encoder-decoder-from",
            type=str,
            metavar="STR",
            help="model to take mt encoder/decoder weights from (for initialization)",
        )
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        decoder_embed_tokens = cls.build_embedding(args, task.target_dictionary, args.decoder_embed_dim)
        encoder = cls.build_encoder(args, task.target_dictionary, decoder_embed_tokens)
        decoder = cls.build_decoder(args, task.target_dictionary, decoder_embed_tokens)

        mt_pretraining_path = getattr(args, "load_pretrained_mt_encoder_decoder_from", None)
        if mt_pretraining_path is not None and Path(mt_pretraining_path).exists():
            mt_state = checkpoint_utils.load_checkpoint_to_cpu(mt_pretraining_path)
            mt_encoder_state_dict = OrderedDict()
            mt_decoder_state_dict = OrderedDict()
            for key in mt_state["model"].keys():
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    subkey = subkey.replace('layers','transformer_layers')
                    mt_encoder_state_dict[subkey] = mt_state["model"][key]
                if key.startswith("decoder"):
                    subkey = key[len("decoder") + 1 :]
                    mt_decoder_state_dict[subkey] = mt_state["model"][key]

            encoder.load_state_dict(mt_encoder_state_dict, strict=False)
            decoder.load_state_dict(mt_decoder_state_dict, strict=False)
            logger.info(f"loaded pretrained mt encoder and decoder from: {mt_pretraining_path}")
        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, dict, embed_tokens):
        encoder = MoyuNetEncoder(args, dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder    
    
    @classmethod
    def build_decoder(cls, args, dict, embed_tokens):
        return MoyuNetDecoder(args, dict, embed_tokens)

@register_model_architecture('moyunet', 'moyunet')
def base_architecture_Moyunet(args):
    args.using_attn = getattr(args, "using_attn", False)
    args.hubert_model_path = getattr(args, "hubert_model_path", None)
    args.freeze_hubert = getattr(args, "freeze_hubert", False) # default is false, 'store_true'
    args.use_asr_finetune_hubert = getattr(args, "use_asr_finetune_hubert", False)
    base_architecture(args)
