import torch
import torch.nn as nn
import torchvision
from src.VIT.config import LOAD_PATH

# Hyperparameters
patch_size: int= 16
embedding_dim: int= 768
hidden_dim: int= embedding_dim*4
n_channels: int= 3
num_heads: int= 12
num_encoders: int= 12
dropout: float= 0.0
num_classes: int= 2
size: int= 224
batch_size: int= 32


class PatchEmbedding(nn.Module):

    """
    Turns a 2D input image into a 1D learnable embedding vector.

    Args:
        in_channels (int): number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert the input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768 (3*16*16).
    """

    def __init__(self,
                 in_channels: int=n_channels,
                 patch_size: int=patch_size,
                 embedding_dim: int=embedding_dim) -> None:
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
        assert_msg = f"Input image size must be divisable by the patch size, image shape: {img_resolution}"
        assert img_resolution % patch_size == 0, assert_msg

        # Forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        # Make sure the output shape has the correct order
        return x_flattened.permute(0, 2, 1)




class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float=0.00):
        """
        Uses multi head attention and multi layer perceptron on the input to get the attentions.
        Args:
            embedding_dim (int): Dimensionality of input and attention feature vectors.
            hidden_dim (int): Dimensionality of hidden layer in feed-forward netword.
            num_heads(int): Number of heads to use in the Multi-Head Attention block.
            dropout (float): Amount of dropout to apply in the feed-forward network.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Attention
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, dropout)

        # Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout))

    def forward(self, x):
        inp_x = self.layer_norm(x)
        x = x + self.mha(inp_x, inp_x, inp_x)[0]
        return x + self.mlp(self.layer_norm(x))



class VisionTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim : int=embedding_dim,
        hidden_dim: int=hidden_dim,
        num_channels: int=n_channels,
        num_heads: int=num_heads,
        num_layers: int=num_encoders,
        num_classes: int=num_classes,
        patch_size: int=patch_size,
        num_patches: int=(size * size) // patch_size**2,
        dropout: float=0.02):
        """
        A paper implementation of the VisionTransformer in:

        "AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"
        url=https://arxiv.org/abs/2010.11929

        Args:
            embedding_dim (int): Dimensionality of the input feature vectors to the Transformer.
            hidden_dim (int): Dimensionality of the hidden layer in the feed-forward networks within the Transformer.
            num_channels (int): Number of channels of the input (3 for RGB or 1 for grayscale).
            num_heads (int): Number of heads to use in the Multi-Head Attention block.
            num_layers (int): Number of layers to use in the Transformer.
            num_classes (int): Number of classes to predict.
            patch_size (int): Number of pixels that the patches have per dimension.
            num_patches (int): Maximum number of patches an image can have.
            dropout (float): Amount of dropout to apply in the feed-forward network and on the input encoding.
        """
        super().__init__()

        # Parameters/Embeddings
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embedding_dim))
        self.patch_embedding = PatchEmbedding()

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embedding_dim)

        self.transformer = nn.Sequential(
            *(AttentionBlock(embedding_dim,
                             hidden_dim,
                             num_heads,
                             dropout=dropout) for _ in range(num_layers)))

        self.mlp_head = nn.Sequential(nn.LayerNorm(embedding_dim),
                                      nn.Linear(embedding_dim, num_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Preprocess input
        x = self.patch_embedding(x)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transformer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        return self.mlp_head(x[0])



class VIT_b16(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super(VIT_b16, self).__init__()
        weights = torchvision.models.ViT_L_16_Weights
        num_classes: int=2
        self.model = torchvision.models.vit_l_16(weights=weights)

        for params in self.model.parameters():
            params.requires_grad=False

        self.model.heads = nn.Sequential(nn.Linear(self.model.hidden_dim, num_classes))

        if pretrained:
            if torch.cuda.is_available():
                state_dict = torch.load(LOAD_PATH)
            else:
                state_dict = torch.load(LOAD_PATH, map_location=torch.device('cpu'))
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '')] = state_dict.pop(key)
            self.model.load_state_dict(state_dict=state_dict)

    def forward(self, x):
        return self.model(x)

    def _get_last_attention(self, x):
         # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding
        for i, encoder in enumerate(self.model.encoder.layers):
            if i < len(self.model.encoder.layers) - 1:
                x = encoder(x)
            else:
            # return attention of the last block
                x = encoder.ln_1(x)
                att, weights = encoder.self_attention(x, x, x, average_attn_weights=False, need_weights=True)
        return att, weights
