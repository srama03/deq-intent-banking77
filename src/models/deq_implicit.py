import torch
import torch.nn as nn

# true deq-> get grads via autograd 
# implicit bw pass: (I-Jf(z*)^T)v = dL/dz*

# ensure we run imnplicit diff instead of the naive

class DEQImplicitFunction(torch.autograd.Function):
    # passes the z_star through along w eqm map and the solver method. uses ctx as a messenger
    @staticmethod
    def forward(ctx, z_star, eqm_map, solver, max_iters):
        ctx.save_for_backward(z_star, eqm_map)
        ctx.max_iters = max_iters
        ctx.solver = solver
        ctx.layer_params = list(solver.__self__.layer.parameters())
        return z_star
    
    # takes the ctx, unpacks whatever tensors and non tensors are saved to calculate the new grads
    @staticmethod
    def backward(ctx, grad_out):      
        z_star, eqm_map = ctx.saved_tensors
        solver = ctx.solver
        v = solver(z_star, eqm_map, grad_out, ctx.max_iters)
        param_grads = torch.autograd.grad(eqm_map, ctx.layer_params, grad_outputs=v, retain_graph=True)
        for param, grad in zip(ctx.layer_params, param_grads):
            param.grad = grad
        return v, None, None, None


class DEQImplicitModel(nn.Module):
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

    def get_deq_map(self, z, x, padding):
        # helper (1) to get the deq mapping {needed for fw solver + bw implicit pass}
        # o/p: f(z,x)= TransformerLayer(z, mask); mask=padding
        deq_map = self.layer(z, src_key_padding_mask=padding) + x
        return deq_map
        
    def solve_eqm(self, x, padding, trace=False):
        # x: embedded inputs (B,T,D)
        # z: initialised at x=> faster convergence
        with torch.no_grad(): # no storing grads
            trace_points = {0, 1, 2, 5, 10, 20, 30, 60, 100} #debugging
            trace_log = []
            z = x
            iters_used=0
            residual = None
            max_iters = self.max_iters_train if self.training else self.max_iters_eval
            for i in range(max_iters):
                z_next = self.get_deq_map(z, x, padding)
                z_new = (1-self.alpha)*z + self.alpha*z_next # damped to prevent oscillations
                residual = (z_new - z).norm() / (z_new.norm() + 1e-6) # damping=> not step wise movt towards eqm but scaled fixed pt residual
                #debugging
                if trace and i in trace_points:
                    trace_log.append((i, float(residual.detach().cpu())))
                z = z_new
                iters_used+=1
                if residual < self.tol:
                    break           
            self.last_iters=iters_used
            self.last_residual=float(residual.detach().cpu()) if residual is not None else None
            self.last_early_stop = (residual is not None and residual < self.tol)
            if trace:
                print("trace:", trace_log)
                print("final:", {"iters": self.last_iters, "residual": self.last_residual, "early": self.last_early_stop})

        return z


    def forward(self, input_ids, attention_mask, trace=False):
        """
        input_ids: (B, T)
        attention_mask: (B,T)
        returns logits: (B, num_labels)
        !!:
        1. find z* numerically, no storing grads (save memory)
        2. let eqm at z* be fresh var; compute map at this val to get access to jacobian
        3. re-engage autograd: find J_f(z*)^Tv; solve bw implicitly
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
        # solve for eqm
        z = self.solve_eqm(x, padding=padding, trace=trace)
        # re-engage autograd
        z_star = z.detach().requires_grad_(True)
        eqm_map = self.get_deq_map(z_star, x, padding)
        max_iters = self.max_iters_train if self.training else self.max_iters_eval
        # apply implicit diffn
        z_star = DEQImplicitFunction.apply(z_star, eqm_map, self.solve_system, max_iters)
        # mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        summation = (z_star*mask).sum(dim=1)
        total = mask.sum(dim=1).clamp(min=1.0)
        pool = summation/total
        # classify
        logits = self.classifier(pool)

        return logits
    
    def apply_Jt(self, z_star, eqm_map, v):
        # helper (2) to get transpose of the local jacobian (at z*) applied to a vector v (to solve linear system)
        # o/p:  J_f(z_star)^Tv
        # remember to get eqm_map=f(z_star) else things go boom
        if z_star.shape == eqm_map.shape and z_star.shape == v.shape :
            trans_jacob_v = torch.autograd.grad(outputs=eqm_map,
                                                 inputs=z_star, 
                                                 grad_outputs=v,
                                                 retain_graph=True)
            JTv = trans_jacob_v[0]
        else:
            raise ValueError("check shapes! need z_star, eqm_map, v to have same shape [B,T,D]")
        return JTv
    
    def solve_system(self, z_star, eqm_map, grad_z, max_iters):
        # helper (3) to solve (I-JT)v = g; where g=dL/dz*
        # o/p: v
        # pass the correct eqm map else things go boom
        assert grad_z.shape == z_star.shape and eqm_map.shape == z_star.shape
        v = grad_z
        for i in range(max_iters):
            
            JTv = self.apply_Jt(z_star, eqm_map, v)
            v_new = JTv + grad_z
            residual = (v_new - v).norm() / (v_new.norm() + 1e-6)
            v = v_new
            if residual < self.tol:
                break
        return v

        

        



