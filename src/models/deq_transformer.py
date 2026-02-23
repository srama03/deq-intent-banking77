import torch
import torch.nn as nn

class DEQModel(nn.Module):
    def __init__(self,
    vocab_size, 
    max_len,
    tol,
    max_iters_train,
    max_iters_eval,
    alpha, 
    d_model=256, 
    n_heads=4, 
    d_ff=1024, 
    dropout=0.1, 
    num_labels=77):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_len, d_model)
        # f(z, x): 1 encoder layer
        # not exactly true deq impl-- a deq-like weight-tied stack
        self.layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads, 
            dim_feedforward=d_ff, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True,
            norm_first=True)
        self.classifier = nn.Linear(d_model, num_labels)
        # tolerance and other hyperparams
        self.tol = tol
        self.max_iters_train = max_iters_train
        self.max_iters_eval = max_iters_eval
        self.alpha = alpha
        # convergence logging (filled during forward)
        self.last_iters = None
        self.last_residual = None

        

    def fixed_point_solver(self, x, padding):
        # x: embedded inputs (B,T,D)
        # z: initialised at x=> faster convergence
        z = x
        iters_used=0
        residual = None
        max_iters = self.max_iters_train if self.training else self.max_iters_eval
        for i in range(max_iters):
            z_next = self.layer(z, src_key_padding_mask=padding) + x # f(z) uses a single encoder layer; x conditions via initialization (z0=x)
            z_new = (1-self.alpha)*z + self.alpha*z_next # damped to prevent oscillations
            # how much z changed-- need residual < tolerance -> measures closeness to FP
            residual = (z_new - z).norm() / (z_new.norm() + 1e-6) # abs mean scales with embedding magnitude and sequence length distribution, hence relative residual
            z = z_new
            iters_used+=1
            
            if residual < self.tol:
                break
        self.last_iters=iters_used
        self.last_residual=float(residual.detach().cpu()) if residual is not None else None
        return z


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
        # fixed pt solving
        z = self.fixed_point_solver(x, padding=padding )
        # mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        summation = (z*mask).sum(dim=1)
        total = mask.sum(dim=1).clamp(min=1.0)
        pool = summation/total
        # classify
        logits = self.classifier(pool)

        return logits
