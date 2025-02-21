from nets.Attention.CBAM import CBAMBlock
from nets.Attention.SE import SEAttention
from nets.Attention.CAA import CAA
from nets.Attention.ECA import EfficientChannelAttention
from nets.Attention.CPCA import CPCA
from nets.Attention.TripletAttention import TripletAttention
from nets.Attention.ShuffleAttention import ShuffleAttention
from nets.Attention.EMCAD import EMCAM
from nets.Attention.BAM import BAMBlock
from nets.Attention.Biformer import BiLevelRoutingAttention_nchw
from nets.Attention.ParNetAttention import ParNetAttention
from nets.Attention.SGE import SpatialGroupEnhance
from nets.Attention.SK import SKAttention


from nets.Attention.simam import SimAM
import torch.nn as nn
import torch

class AttentionFactory:
    @staticmethod
    def get_attention(attention_type, in_planes, **kwargs):
        if attention_type == "cbam":
            return CBAMBlock(in_planes, **kwargs)
        elif attention_type == "se":
            return SEAttention(in_planes, **kwargs)
        elif attention_type == "caa":
            return CAA(in_planes, **kwargs)
        elif attention_type == "eca":
            return EfficientChannelAttention(in_planes, **kwargs)
        elif attention_type == "cpca":
            return CPCA(in_planes, **kwargs)
        elif attention_type == "ta":
            return TripletAttention(in_planes, **kwargs)
        elif attention_type == "sa":
            return ShuffleAttention(in_planes, **kwargs)
        elif attention_type == "emcam":
            return EMCAM(in_planes, **kwargs)
        elif attention_type == "bam":  # 新增 BAM 选择
            return BAMBlock(channel=in_planes, **kwargs)
        elif attention_type == "bi_level_routing_nchw":  # Bi-Level Routing Attention
            return BiLevelRoutingAttention_nchw(in_planes, **kwargs)
        elif attention_type == "par_net_attention":  # ParNet Attention
            return ParNetAttention(in_planes, **kwargs)
        elif attention_type == "sge":  # Spatial Group Enhance
            return SpatialGroupEnhance(in_planes, **kwargs)
        elif attention_type == "sk":  # SK Attention
            return SKAttention(in_planes, **kwargs)
        elif attention_type == "simam":  # SK Attention
            return SimAM(in_planes, **kwargs)


        else:
            return nn.Identity()  # 默认无注意力机制


