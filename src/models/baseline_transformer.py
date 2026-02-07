import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self,
    vocab_size, 
    max_len, 
    d_model=256, 
    n_heads=4, 
    num_layers=3, 
    d_ff=1024, 
    dropout=0.1, 
    num_labels=77):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads, 
            dim_feedforward=d_ff, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (B, T)
        attention_mask: (B,T)
        returns logits: (B, num_labels)
        """
        B, T = input_ids.shape
        # embeddings
        x = self.token_embeddings(input_ids)
        # position indices
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        # add positional embeddings
        x = x + self.pos_embeddings(positions)
        # padding
        padding = (attention_mask==0)
        # transformer
        x = self.encoder(x, src_key_padding_mask=padding)
        # mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        summation = (x*mask).sum(dim=1)
        total = mask.sum(dim=1).clamp(min=1.0)
        pool = summation/total
        # classify
        logits = self.classifier(pool)

        return logits

        


