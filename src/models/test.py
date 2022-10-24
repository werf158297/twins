import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from functools import partial
import mindspore.numpy as np
import math
from mindspore import Tensor, context, Parameter,Model
from mindspore.common import initializer as weight_init
from mindspore.ops import operations as ops
from .misc import DropPath1D as Droppath
from .misc import Identity
class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(1-drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Cell):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
      #  assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.kv = nn.Dense(dim, dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1-attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1-proj_drop)
        self.softmax = nn.Softmax()

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio,has_bias=True)
            self.norm = nn.LayerNorm([dim])

    def construct(self, x, H, W):
        B, N, C = x.shape
        q=self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0,2,1,3)
        #input_perm = (0, 2, 1, 3)
        #trans=ops.Transpose()
        #q = trans(q1, input_perm)
        
        if self.sr_ratio > 1:
            
            x_ = x.transpose(0,2,1).reshape(B, C, H, W)
            
            
            x_ = self.sr(x_).reshape(B, C, -1).transpose(0,2,1)
            x_ = self.norm(x_)
            
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        else:
           
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        k=k.transpose(0,1,3,2)
 
        attn = np.matmul(q,k)* self.scale
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (np.matmul(attn,v)).transpose(0,2,1,3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer([dim])
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = Droppath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
    
        self.img_size = img_size
        self.patch_size = patch_size
       
        self.H, self.W = img_size // patch_size, img_size // patch_size
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,has_bias=True)
        self.norm = nn.LayerNorm([embed_dim])

    def construct(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)
        B,c, h, w = x.shape
        x=x.reshape(B,c,-1).transpose(0,2,1)
        x = self.norm(x)
        H, W = H // self.patch_size, W // self.patch_size

        return x, (H, W)



class PosCNN(nn.Cell):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 3, s, has_bias=True, group=embed_dim)
        self.s = s

    def construct(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(0,2,1).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        B, c, h,w = x.shape
        x = x.reshape(B,c,-1).transpose(0,2,1)
        return x

    ##def no_weight_decay(self):
        #return ['proj.%d.weight' % i for i in range(4)]


class CPVTV2(nn.Cell):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.patch_embeds = nn.CellList()
        self.pos_drops = nn.CellList()
        self.blocks = nn.CellList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))
            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(embed_dims) - 1 else self.patch_embeds[-1].num_patches
           
            self.pos_drops.append(nn.Dropout(1-drop_rate))
 
        dpr =drop_path_rate/(sum(depths)-1)
        
        for k in range(len(depths)):
            _block = nn.CellList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr*(i+sum(depths[:k])), norm_layer=norm_layer,
                sr_ratio=sr_ratios[k])
                for i in range(depths[k])])
            
            self.blocks.append(_block)
        self.norm = norm_layer([embed_dims[-1]])
        

        self.head = nn.Dense(embed_dims[-1], num_classes) if num_classes > 0 else Identity()

        self.pos_block = nn.CellList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )
        self.init_weights()

    
    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
    def construct(self, x):
        B = x.shape[0]
        

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            if i < len(self.depths) - 1:
               
                x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
        x = self.norm(x)
        x= x.mean(1)  # GAP here
        x = self.head(x)

        return x
   
def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }



def pcpvt_small_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model

def pcpvt_base_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


def pcpvt_large_v0(pretrained=False, **kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':
    
    
    
    
    from mindspore import Tensor
    import numpy 
    from mindspore import context, set_seed, Model
    import matplotlib.pyplot as plt
    from PIL import Image
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

 
    def visualize_model(path):
	    image = Image.open(path).convert("RGB")
	    image = image.resize((224, 224))
	    plt.imshow(image)

    # 归一化处理
	    mean = numpy.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
	    std = numpy.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
	    image = numpy.array(image)
	    image = (image - mean) / std
	    image = image.astype(numpy.float32)

    # 图像通道由(h, w, c)转换为(c, h, w)
	    image = numpy.transpose(image, (2, 0, 1))

    # 扩展数据维数为(1, c, h, w)
	    image = numpy.expand_dims(image, axis=0)

    # 定义并加载网络
	    net = pcpvt_small_v0()
    #param_dict = load_checkpoint("./best.ckpt")
    #load_param_into_net(net, param_dict)
	    model = Model(net)

	    

    # 模型预测
	    pre = model.predict(Tensor(image))
	    print(pre.shape)
	    result = numpy.argmax(pre)
    
	    result=1 if result>499 else 0
	    class_name = {0: "cat", 1: "dog"}
	    plt.title(f"Predict: {class_name[result]}")
	    return result

    image1 = "./test.jpeg"
    plt.figure(figsize=(7, 7))

    visualize_model(image1)
    plt.show()
