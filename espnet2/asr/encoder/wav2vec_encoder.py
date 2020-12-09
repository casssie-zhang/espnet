# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet2.asr.encoder.abs_encoder import AbsEncoder
import fairseq
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model

def get_output_lens(conv_layers, input_lens):
    out = input_lens
    for layer in conv_layers:
        conv_layer = layer[0]
        kernel_size, stride = conv_layer.kernel_size, conv_layer.stride
        out = (out-kernel_size[0]) // stride[0] + 1
    return out

class Wav2vecTransformerEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size,
        output_size: int = 768,
        dropout_rate: float = 0.1,
        model_path="/home/ubuntu/project/manifest/train/outputs/2020-12-04/06-26-12/checkpoints/checkpoint_best.pt",
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        print("Wav2Vec encoder successfully loaded!")

        model = model[0]
        if type(model) == Wav2VecCtc:
            model = model.w2v_encoder.w2v_model
        elif type(model) == Wav2Vec2Model:
            pass

        model.feature_grad_mult = 0  # zero grad
        print("Conv feature extraction has been freezed.")


        self.wav2vec = model
        self._output_size = output_size

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        self.wav2vec.feature_grad_mult = 0 # make sure conv feature extraction has been freezed
        xs_pad = self.wav2vec.forward(xs_pad, mask=False, features_only=True)['x']
        print(xs_pad[0,:,1])
        feats_lens = []
        for lens in ilens:
            feats_lens.append(get_output_lens(self.wav2vec.feature_extractor.conv_layers, lens))
        olens = torch.stack(feats_lens)

        # xs_pad = self.projection(xs_pad)

        return xs_pad, olens, None
