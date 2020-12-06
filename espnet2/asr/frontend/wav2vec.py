# import copy
from typing import Optional
from typing import Tuple
# from typing import Union

# import humanfriendly
# import numpy as np
import torch
# from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
# from espnet2.layers.log_mel import LogMel
# from espnet2.layers.stft import Stft
# from espnet2.utils.get_default_kwargs import get_default_kwargs
import fairseq
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model

def get_output_lens(wav2vec_model, input_lens):
    out = input_lens
    for layer in wav2vec_model.feature_extractor.conv_layers:
        conv_layer = layer[0]
        kernel_size, stride = conv_layer.kernel_size, conv_layer.stride
        out = (out-kernel_size[0]) // stride[0] + 1
    return out


class Wav2vecFrontend(AbsFrontend):
    """Wav2vec frontend structure for ASR.
    """

    def __init__(
        self,
        model_path="/home/ubuntu/project/manifest/train/outputs/2020-12-04/06-26-12/checkpoints/checkpoint_best.pt",
        embedding_dim:int=768):
        """
        Args:
            model_path: wav2vec model path
                without finetuning:  /home/ubuntu/project/model/wav2vec_small.pt
                finetuning best checkpoint:  /home/ubuntu/project/manifest/train/outputs/2020-12-04/06-26-12/checkpoints/checkpoint_best.pt
            embedding_dim: model output features dim
        """
        assert check_argument_types()
        super().__init__()
        
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        print("Wav2Vec model successfully loaded!")

        model = model[0]
        if type(model) == Wav2VecCtc:
            model = model.w2v_encoder.w2v_model
        elif type(model) == Wav2Vec2Model:
            pass

        # freeze layers:
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
        print("Setting all Wav2vec layers parameters requires_grad=False")

        self.wav2vec = model
        self.embedding_dim = embedding_dim

    def output_size(self) -> int:
        return self.embedding_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        input_feats = self.wav2vec.forward(input, mask=False, features_only=True)['x']
        input_feats = input_feats.detach()
        feats_lens = []
        for lens in input_lengths:
            feats_lens.append(get_output_lens(self.wav2vec, lens))
        feats_lens = torch.stack(feats_lens)

        return input_feats, feats_lens
