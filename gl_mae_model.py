import torch
import torch.nn as nn
import numpy as np
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


# PatchShuffle类，用于对patches进行随机打乱
class PatchShuffle(nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes



class GL_MAE_Encoder(nn.Module):
    def __init__(self,
                 input_channels=12,
                 image_size=100,
                 emb_dim=192,
                 num_layer=6,
                 num_head=3,
                 mask_ratio=0.75) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.num_patches_global = 25
        self.num_patches_mid = 100
        self.num_patches_local = 400

        self.pos_embedding_global = nn.Parameter(torch.zeros(self.num_patches_global, 1, emb_dim))
        self.pos_embedding_mid = nn.Parameter(torch.zeros(self.num_patches_mid, 1, emb_dim))
        self.pos_embedding_local = nn.Parameter(torch.zeros(self.num_patches_local, 1, emb_dim))

        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify_global = nn.Conv2d(input_channels, emb_dim, kernel_size=(4, 100), stride=(4, 100))
        self.patchify_mid = nn.Conv2d(input_channels, emb_dim, kernel_size=(1, 100), stride=(1, 100))
        self.patchify_local = nn.Conv2d(input_channels, emb_dim, kernel_size=(1, 25), stride=(1, 25))

        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)

        # 添加跨层对齐模块
        # self.cross_level_alignment = CrossLevelAlignment(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding_global, std=.02)
        trunc_normal_(self.pos_embedding_local, std=.02)
        trunc_normal_(self.pos_embedding_mid, std=.02)

    def forward(self, img):
        # patches = self.patchify(img)
        # patches = rearrange(patches, 'b c h w -> (h w) b c')
        # patches = patches + self.pos_embedding
        patches_global = self.patchify_global(img)
        patches_mid = self.patchify_mid(img)
        patches_local = self.patchify_local(img)

        patches_global = rearrange(patches_global, 'b c h w -> (h w) b c')
        patches_mid = rearrange(patches_mid, 'b c h w -> (h w) b c')
        patches_local = rearrange(patches_local, 'b c h w -> (h w) b c')

        patches_global = patches_global + self.pos_embedding_global
        patches_mid = patches_mid + self.pos_embedding_mid
        patches_local = patches_local + self.pos_embedding_local

        # patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches_global, forward_indexes_global, backward_indexes_global = self.shuffle(patches_global)
        patches_mid, forward_indexes_mid, backward_indexes_mid = self.shuffle(patches_mid)
        patches_local, forward_indexes_local, backward_indexes_local = self.shuffle(patches_local)

        # patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        # patches = rearrange(patches, 't b c -> b t c')
        # features = self.layer_norm(self.transformer(patches))
        # features = rearrange(features, 'b t c -> t b c')
        patches_global = torch.cat([self.cls_token.expand(-1, patches_global.shape[1], -1), patches_global], dim=0)
        patches_mid = torch.cat([self.cls_token.expand(-1, patches_mid.shape[1], -1), patches_mid], dim=0)
        patches_local = torch.cat([self.cls_token.expand(-1, patches_local.shape[1], -1), patches_local], dim=0)

        patches_global = rearrange(patches_global, 't b c -> b t c')
        patches_mid = rearrange(patches_mid, 't b c -> b t c')
        patches_local = rearrange(patches_local, 't b c -> b t c')

        features_global = self.layer_norm(self.transformer(patches_global))
        features_mid = self.layer_norm(self.transformer(patches_mid))
        features_local = self.layer_norm(self.transformer(patches_local))

        features_global = rearrange(features_global, 'b t c -> t b c')
        features_mid = rearrange(features_mid, 'b t c -> t b c')
        features_local = rearrange(features_local, 'b t c -> t b c')

        # features_global, features_mid, features_local = self.cross_level_alignment(features_global, features_mid,
        #                                                                            features_local)

        # return features, backward_indexes
        return features_global, features_mid, features_local, backward_indexes_global, backward_indexes_mid, backward_indexes_local


class GL_MAE_Decoder(nn.Module):
    def __init__(self,
                 input_channels=12,
                 image_size=100,
                 emb_dim=192,
                 # num_layer=4,
                 num_layer=16,
                 num_head=3) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.num_patches_global = 25
        self.num_patches_mid = 100
        self.num_patches_local = 400

        self.pos_embedding_global = nn.Parameter(torch.zeros(self.num_patches_global + 1, 1, emb_dim))
        self.pos_embedding_mid = nn.Parameter(torch.zeros(self.num_patches_mid + 1, 1, emb_dim))
        self.pos_embedding_local = nn.Parameter(torch.zeros(self.num_patches_local + 1, 1, emb_dim))

        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head_global = nn.Linear(emb_dim, input_channels * 4 * 100)
        self.head_mid = nn.Linear(emb_dim, input_channels * 1 * 100)
        self.head_local = nn.Linear(emb_dim, input_channels * 1 * 25)

        self.patch2img_global = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=4, p2=100,
                                          h=image_size // 4)
        self.patch2img_mid = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=1, p2=100,
                                       h=image_size // 1)
        self.patch2img_local = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=1, p2=25,
                                         h=image_size // 1)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding_global, std=.02)
        trunc_normal_(self.pos_embedding_mid, std=.02)
        trunc_normal_(self.pos_embedding_local, std=.02)

    def forward(self, features_global, features_mid, features_local, backward_indexes_global, backward_indexes_mid,
                backward_indexes_local):
        T_global = features_global.shape[0]
        T_mid = features_mid.shape[0]
        T_local = features_local.shape[0]

        backward_indexes_global = torch.cat(
            [torch.zeros(1, backward_indexes_global.shape[1]).to(backward_indexes_global), backward_indexes_global + 1],
            dim=0)
        backward_indexes_mid = torch.cat(
            [torch.zeros(1, backward_indexes_mid.shape[1]).to(backward_indexes_mid), backward_indexes_mid + 1], dim=0)
        backward_indexes_local = torch.cat(
            [torch.zeros(1, backward_indexes_local.shape[1]).to(backward_indexes_local), backward_indexes_local + 1],
            dim=0)

        features_global = torch.cat(
            [features_global, self.mask_token.expand(backward_indexes_global.shape[0] - features_global.shape[0],
                                                     features_global.shape[1], -1)],
            dim=0)
        features_mid = torch.cat(
            [features_mid,
             self.mask_token.expand(backward_indexes_mid.shape[0] - features_mid.shape[0], features_mid.shape[1], -1)],
            dim=0)
        features_local = torch.cat(
            [features_local,
             self.mask_token.expand(backward_indexes_local.shape[0] - features_local.shape[0], features_local.shape[1],
                                    -1)],
            dim=0)

        features_global = take_indexes(features_global, backward_indexes_global)
        features_mid = take_indexes(features_mid, backward_indexes_mid)
        features_local = take_indexes(features_local, backward_indexes_local)

        features_global = features_global + self.pos_embedding_global
        features_mid = features_mid + self.pos_embedding_mid
        features_local = features_local + self.pos_embedding_local

        features_global = rearrange(features_global, 't b c -> b t c')
        features_mid = rearrange(features_mid, 't b c -> b t c')
        features_local = rearrange(features_local, 't b c -> b t c')

        features_global = self.transformer(features_global)
        features_mid = self.transformer(features_mid)
        features_local = self.transformer(features_local)

        features_global = rearrange(features_global, 'b t c -> t b c')
        features_mid = rearrange(features_mid, 'b t c -> t b c')
        features_local = rearrange(features_local, 'b t c -> t b c')

        features_global = features_global[1:]  # remove global feature
        features_mid = features_mid[1:]
        features_local = features_local[1:]

        patches_global = self.head_global(features_global)
        patches_mid = self.head_mid(features_mid)
        patches_local = self.head_local(features_local)

        mask_global = torch.zeros_like(patches_global)
        mask_mid = torch.zeros_like(patches_mid)
        mask_local = torch.zeros_like(patches_local)

        mask_global[T_global - 1:] = 1
        mask_mid[T_mid - 1:] = 1
        mask_local[T_local - 1:] = 1

        mask_global = take_indexes(mask_global, backward_indexes_global[1:] - 1)
        mask_mid = take_indexes(mask_mid, backward_indexes_mid[1:] - 1)
        mask_local = take_indexes(mask_local, backward_indexes_local[1:] - 1)

        img_global = self.patch2img_global(patches_global)
        img_mid = self.patch2img_mid(patches_mid)
        img_local = self.patch2img_local(patches_local)

        mask_global = self.patch2img_global(mask_global)
        mask_mid = self.patch2img_mid(mask_mid)
        mask_local = self.patch2img_local(mask_local)

        return img_global, img_mid, img_local, mask_global, mask_mid, mask_local


# MAE_ViT类，整合MAE的编码器和解码器
class GL_MAE_ViT(nn.Module):
    def __init__(self,
                 input_channels=12,
                 image_size=100,
                 emb_dim=192,
                 encoder_layer=6,
                 encoder_head=3,
                 # decoder_layer=4,
                 decoder_layer=16,
                 decoder_head=3,
                 mask_ratio=0.75) -> None:
        super().__init__()
        self.encoder = GL_MAE_Encoder(input_channels, image_size, emb_dim, encoder_layer,
                                      encoder_head, mask_ratio)
        self.decoder = GL_MAE_Decoder(input_channels, image_size, emb_dim, decoder_layer,
                                      decoder_head)

    def forward(self, img):
        features_global, features_mid, features_local, backward_indexes_global, backward_indexes_mid, backward_indexes_local = self.encoder(
            img)
        predicted_img_global, predicted_img_mid, predicted_img_local, mask_global, mask_mid, mask_local = self.decoder(
            features_global, features_mid, features_local, backward_indexes_global, backward_indexes_mid,
            backward_indexes_local)
        return predicted_img_global, predicted_img_mid, predicted_img_local, mask_global, mask_mid, mask_local


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: GL_MAE_Encoder, num_classes=6) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding_global = encoder.pos_embedding_global
        self.pos_embedding_mid = encoder.pos_embedding_mid
        self.pos_embedding_local = encoder.pos_embedding_local

        self.patchify_global = encoder.patchify_global
        self.patchify_mid = encoder.patchify_mid
        self.patchify_local = encoder.patchify_local

        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        # self.head_global = torch.nn.Linear(self.pos_embedding_global.shape[-1], num_classes)
        # self.head_mid = torch.nn.Linear(self.pos_embedding_mid.shape[-1], num_classes)
        # self.head_local = torch.nn.Linear(self.pos_embedding_local.shape[-1], num_classes)
        # self.head = torch.nn.Linear(
        #     self.pos_embedding_global.shape[-1] +
        #     self.pos_embedding_mid.shape[-1] +
        #     self.pos_embedding_local.shape[-1],
        #     num_classes
        # )

        # 定义1D卷积层，用于融合特征
        self.fc_fusion = torch.nn.Linear(192 * 3, 192)

        # 最终分类头
        self.fc = torch.nn.Linear(192, num_classes)

    def forward(self, x):
        patches_global = self.patchify_global(x)
        patches_mid = self.patchify_mid(x)
        patches_local = self.patchify_local(x)

        patches_global = rearrange(patches_global, 'b c h w -> (h w) b c')
        patches_mid = rearrange(patches_mid, 'b c h w -> (h w) b c')
        patches_local = rearrange(patches_local, 'b c h w -> (h w) b c')

        patches_global = patches_global + self.pos_embedding_global
        patches_mid = patches_mid + self.pos_embedding_mid
        patches_local = patches_local + self.pos_embedding_local

        patches_global = torch.cat([self.cls_token.expand(-1, patches_global.shape[1], -1), patches_global], dim=0)
        patches_mid = torch.cat([self.cls_token.expand(-1, patches_mid.shape[1], -1), patches_mid], dim=0)
        patches_local = torch.cat([self.cls_token.expand(-1, patches_local.shape[1], -1), patches_local], dim=0)

        patches_global = rearrange(patches_global, 't b c -> b t c')
        patches_mid = rearrange(patches_mid, 't b c -> b t c')
        patches_local = rearrange(patches_local, 't b c -> b t c')

        features_global = self.layer_norm(self.transformer(patches_global))
        features_mid = self.layer_norm(self.transformer(patches_mid))
        features_local = self.layer_norm(self.transformer(patches_local))

        features_global = rearrange(features_global, 'b t c -> t b c')
        features_mid = rearrange(features_mid, 'b t c -> t b c')
        features_local = rearrange(features_local, 'b t c -> t b c')

        # 拼接全局、中级和局部特征
        combined_features = torch.cat([
            features_global[0],
            features_mid[0],
            features_local[0]
        ], dim=-1)

        # 使用CNN融合特征
        combined_features = rearrange(combined_features, 'b c -> b 1 c')  # 添加一个虚拟的维度用于卷积操作
        # 使用全连接层融合特征
        fused_features = self.fc_fusion(combined_features)  # (8, 1, 192)
        fused_features = fused_features.squeeze(1)  # (8, 192)

        # logits = self.head(features[0])
        logits = self.fc(fused_features)
        return logits


if __name__ == '__main__':
    x = torch.rand(8, 12, 100, 100)  # 输入的时序信号
    # 实例化并使用CViT
    # model = GL_MAE_ViT()
    # # predicted_img, mask = model(x)
    # predicted_img_global, predicted_img_mid, predicted_img_local, mask_global, mask_mid, mask_local = model(x)
    # # print(predicted_img.shape, mask.shape)
    # print(predicted_img_global.shape, predicted_img_mid.shape, predicted_img_local.shape)
    model = ViT_Classifier(GL_MAE_ViT().encoder, num_classes=6)
    logits = model(x)
    print(logits.shape)