import einops
from tqdm.notebook import tqdm

from torchinfo import summary


import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

patch_size = 16
latent_size = 768
n_channels = 3
num_heads = 12
num_encoders = 12
dropout = 0.1
num_classes = 2
size = 224

epochs = 10
base_lr = 10e-3
weight_decay = 0.03
batch_size = 32


# # Implementation of input linear projection
# class InputEmbedding(nn.Module):
#     def __init__(self, patch_size=patch_size, n_channels=n_channels, device=device, latent_size=latent_size, batch_size= batch_size):
#         super(InputEmbedding, self).__init__()
#         self.latent_size = latent_size
#         self.patch_size = patch_size
#         self.n_channels = n_channels
#         self.device = device
#         self.batch_size = batch_size
#         self.input_size = self.patch_size*self.patch_size*self.n_channels

#         # Linear projection
#         self.linearProjection = nn.Linear(self.input_size, self.latent_size)

#         # Class token
#         self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

#         # Positional embedding
#         self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

#     def forward(self, input_data):
#         input_data = input_data.to(device)

#         # Patchify input image
#         patches = einops.rearrange(
#             input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
#         linear_projection = self.linearProjection(patches).to(device)
#         b, n, _ = linear_projection.shape
#         linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
#         pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b m d', m = n+1)

#         linear_projection += pos_embed

#         return linear_projection
    


class PatchEmbedding(nn.Module):

    """
    Turns a 2D input image into a 1D learnable embedding vector.

    Args:
        in_channels (int): number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert the input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768 (3*16*16)
    """

    def __init__(self,
                 in_channels: int=n_channels,
                 patch_size: int=patch_size,
                 embedding_dim: int=latent_size,
                 device=device) -> None:
        super(PatchEmbedding, self).__init__()

        # Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,  # By making the kernel_size and stride both equal the patch_size
                                # -> the convolution will turn the image into the specified patch_size
            padding=0)
        
        # Create a layer to flatten the patch feature maps into a single dimension.
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)


    def forward(self, x): 
        # Create an assertion to check that the inputs are the correct shape
        img_resolution = x.shape[-1]
        assert img_resolution % patch_size == 0, f"Input image size must be divisable by the patch size, image shape: {img_resolution}"

        # Forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        # Make sure the output shape has the correct order
        return x_flattened.permute(0, 2, 1)
    


class EncoderBlock(nn.Module):
    def __init__(self, latent_size=latent_size, num_heads=num_heads, device=device, dropout=dropout):
        super(EncoderBlock, self).__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device
        self.dropout = dropout

        # Normalization
        self.norm = nn.LayerNorm(self.latent_size)

        self.multihead = nn.MultiheadAttention(
            self.latent_size, self.num_heads, dropout=self.dropout
            )
        
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size*4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size*4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, embedded_patches):

        firstnorm_out = self.norm(embedded_patches)
        attention_out, attention_matrix = self.multihead(firstnorm_out, firstnorm_out, firstnorm_out)

        # First residual connection
        first_added = attention_out + embedded_patches

        secondnorm_out = self.norm(first_added)
        ff_out = self.enc_MLP(secondnorm_out)

        return ff_out + first_added, attention_matrix



class VisionTransformer(nn.Module):
    def __init__(self, num_encoders=num_encoders, latent_size=latent_size, device=device, num_classes=num_classes, dropout=dropout):
        super(VisionTransformer, self).__init__()

        self.num_encoders = num_encoders
        self.latent_size = latent_size
        self.device = device
        self.num_classes = num_classes
        self.dropout = dropout

        self.embedding = PatchEmbedding()

        # Create the stack of encoders
        self.encStack = nn.ModuleList(
            [EncoderBlock() for i in range(self.num_encoders)]
            )

        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes)
            )

    def forward(self, input):

        # if input.dim() == 3:
        #     input.unsqueeze(0)

        enc_out = self.embedding(input)
        attention_matrices = []

        for enc_layer in self.encStack:
            enc_out, attention_matrix = enc_layer(enc_out)
            attention_matrices.append(attention_matrix)


        cls_token_embed = enc_out[:, 0]

        # if input.dim() == 3:
        #     cls_token_embed = cls_token_embed.squeeze(0)

        return self.MLP_head(cls_token_embed), attention_matrices
    




