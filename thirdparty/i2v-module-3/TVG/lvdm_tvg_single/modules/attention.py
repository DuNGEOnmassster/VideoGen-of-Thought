import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
from lvdm_tvg_single.common import (
    checkpoint,
    exists,
    default,
)
from lvdm_tvg_single.basics import zero_module


def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    dist_sq = torch.cdist(X1 / lengthscale, X2 / lengthscale, p=2).pow(2)
    return variance * torch.exp(-0.5 * dist_sq)

@torch.no_grad()
class GaussianProcess:
    def __init__(self, kernel, X_train, y_train, noise_var=1e-4):#torch.Size([4096, 512])
        self.kernel = kernel
        self.X_train = X_train
        self.y_train = y_train
        self.noise_var = noise_var
        self.K = self.kernel(self.X_train, self.X_train) + noise_var * torch.eye(len(X_train)).to(X_train.device)


    def predict(self, X_new):#torch.Size([15, 4096, 320])
        batch_size, num_features, feature_dims = X_new.shape

        # Rearrange X_new for batch processing
        X_new = rearrange(X_new, 'b f d -> (b f) d')#torch.Size([13, 4096, 320])

        # Calculate kernels
        K_star = self.kernel(self.X_train, X_new)  # Shape: (num_train, batch_size*num_features) ([4096, 320]),([53248, 320])->([4096, 53248])
        K_star = rearrange(K_star, 'n (b f) -> b f n', b=batch_size, f=num_features)  # Reshape K_star torch.Size([13, 4096, 4096])
        # K_star_star = self.kernel(X_new, X_new)  # Shape: (batch_size*num_features, batch_size*num_features)

        # K_star_star = rearrange(K_star_star, '(b1 f1) (b2 f2) -> b1 f1 b2 f2', b1=batch_size, f1=num_features, b2=batch_size, f2=num_features)
        # Take mean over the batch dimension
        # K_star_star = K_star_star.mean(dim=0)

        # Compute predictive mean and covariance 
        K_inv = torch.inverse(self.K)#torch.Size([4096, 4096])->torch.Size([4096, 4096])->torch.Size([4096, 4096])
        mu = torch.matmul(K_star.transpose(1, 2), torch.matmul(K_inv, self.y_train))  # Shape: (batch_size, num_features, feature_dims)
        # cov = K_star_star.transpose(0,1) - torch.matmul(K_star.transpose(1, 2), torch.matmul(K_inv, K_star))  # Shape: (batch_size, num_features, num_features)

        return mu.contiguous()

@torch.no_grad()
def slerp(latents0, latents1, fract_mixing, adain=False):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """
    if latents0 is None or latents1 is None:
        return latents0 if latents1 is None else latents1
    p0 = latents0
    p1 = latents1
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    p0 = p0.double()
    p1 = p1.double()

    # if adain:
    #     mean1, std1 = calc_mean_std(p0)
    #     mean2, std2 = calc_mean_std(p1)
    #     mean = mean1 * (1 - fract_mixing) + mean2 * fract_mixing
    #     std = std1 * (1 - fract_mixing) + std2 * fract_mixing
        
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1

    # if adain:
    #     interp = F.batch_norm(interp,mean,std,) * std + mean

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
    latents0=interp

    return latents0

class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., 
                 relative_position=False, temporal_length=None, video_length=None, image_cross_attention=False, image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        
        self.relative_position = relative_position
        if self.relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.text_context_len = text_context_len
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            if image_cross_attention_scale_learnable:
                self.register_parameter('alpha', nn.Parameter(torch.tensor(0.)) )


    def forward(self, x, context=None, mask=None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            # else:    
                    
            #         num_frames=context.shape[1]

            #         half_size = context.shape[0] // 2
            #         x_first_half = context[ :half_size,:, :]
            #         x_second_half = context[ half_size:,:, :]
                


            #         alpha_list = torch.linspace(1, 0, num_frames).view(1,-1,1).to(x.device)
            #         new_first_half = alpha_list*x_first_half + (1-alpha_list)* x_second_half
            #         new_second_half = alpha_list*x_second_half + (1-alpha_list)* x_first_half
            #         context=torch.cat([new_first_half,new_second_half],dim=0)

            k = self.to_k(context)
            v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
            sim += sim2
        del k

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask>0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)


        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)


        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
            else:
                out = out + self.image_cross_attention_scale * out_ip
        
        return self.to_out(out)
    
    def efficient_forward(self, x, context=None, mask=None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
        else:
            if not spatial_self_attn:
                context = context[:,:self.text_context_len,:]
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        
        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = (
                out_ip.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
            else:
                out = out + self.image_cross_attention_scale * out_ip
           
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                disable_self_attn=False, attention_cls=None, video_length=None, image_cross_attention=False, image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, video_length=video_length, image_cross_attention=image_cross_attention, image_cross_attention_scale=image_cross_attention_scale, image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,text_context_len=text_context_len)
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint


    def forward(self, x, context=None, mask=None, **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)


    def _forward(self, x, context=None, mask=None):
        _x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + 0.9*x
        if _x.shape[0]<_x.shape[1]:
            gp = GaussianProcess(kernel=rbf_kernel, X_train=x[0,:,:], y_train=x[-1,:,:])
            num_frames=_x.shape[0]
            x[1:num_frames-2,:,:] = gp.predict(x[1:num_frames-2,:,:])
            # for f_idx in range(1,num_frames-2):
            #     x[f_idx,:,:], _ = gp.predict(x[f_idx,:,:])
        else:
            gp = GaussianProcess(kernel=rbf_kernel, X_train=x[:,0,:], y_train=x[:,-1,:])
            num_frames=_x.shape[1]
            tmp_x = gp.predict(x[:,1:num_frames-2,:].transpose(1,0))
            x[:,1:num_frames-2,:] = tmp_x.transpose(1,0)
            # for f_idx in range(1,num_frames-2):
            #     x[:,f_idx,:], _ = gp.predict(x[:,f_idx,:])
            num_frames=_x.shape[1]
        x = _x + 0.1*x

        _x = self.attn2(self.norm2(x), context=context, mask=mask) + 0.9*x

        if _x.shape[0]<_x.shape[1]:
            gp = GaussianProcess(kernel=rbf_kernel, X_train=x[0,:,:], y_train=x[-1,:,:])
            num_frames=_x.shape[0]
            x[1:num_frames-2,:,:]= gp.predict(x[1:num_frames-2,:,:])
            # for f_idx in range(1,num_frames-2):
            #     x[f_idx,:,:], _ = gp.predict(x[f_idx,:,:])
        else:
            gp = GaussianProcess(kernel=rbf_kernel, X_train=x[:,0,:], y_train=x[:,-1,:])
            num_frames=_x.shape[1]
            tmp_x = gp.predict(x[:,1:num_frames-2,:].transpose(1,0))
            x[:,1:num_frames-2,:] = tmp_x.transpose(1,0)
            # for f_idx in range(1,num_frames-2):
            #     x[:,f_idx,:], _ = gp.predict(x[:,f_idx,:])
        x = _x + 0.1*x

        x = self.ff(self.norm3(x)) + x
        return x

    # def _forward2(self, x, context=None, mask=None):
    #     weights = 0.9
    #     _x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x

        
    #     if _x.shape[0]<_x.shape[1]:
    #         # num_frames=_x.shape[0]


    #         # half_size = _x.shape[0] // 2
    #         # x_first_half = x[:half_size]
    #         # x_second_half = x[half_size:]



    #         # num_frames=_x.shape[0]


    #         half_size = _x.shape[0] // 2
    #         num_frames=half_size
    #         x_first_half = x[:half_size]
    #         x_second_half = x[half_size:]

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[0,:,:], y_train=x_first_half[-1,:,:])
    #         # x_first_half[1:num_frames-2,:,:] = gp.predict(x_first_half[1:num_frames-2,:,:])

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[0,:,:], y_train=x_second_half[-1,:,:])
    #         # x_second_half[1:num_frames-2,:,:] = gp.predict(x_second_half[1:num_frames-2,:,:])

    #         gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[0,:,:], y_train=x_first_half[-1,:,:])
    #         x_first_half[1:num_frames-2,:,:], cov_first = gp.predict(x_first_half[1:num_frames-2,:,:])

    #         gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[0,:,:], y_train=x_second_half[-1,:,:])
    #         x_second_half[1:num_frames-2,:,:], cov_second = gp.predict(x_second_half[1:num_frames-2,:,:])

    #         #时序传播
    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[0,:,:], y_train=x_first_half[1,:,:])
    #         # # temp_x1 = x_first_half[0:num_frames-3,:,:]
    #         # x_first_half[1:num_frames-2,:,:] = gp.predict(x_first_half[0:num_frames-3,:,:])

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[0,:,:], y_train=x_second_half[1,:,:])
    #         # x_second_half[1:num_frames-2,:,:] = gp.predict(x_second_half[0:num_frames-3,:,:])

    #         weight_first = torch.sqrt(cov_first.diagonal(dim1=-2, dim2=-1)).mean(-1)
    #         weight_second = torch.sqrt(cov_second.diagonal(dim1=-2, dim2=-1)).mean(-1)

    #         # 反序操作
    #         x_first_half_reversed = torch.flip(x_first_half, dims=[0])
    #         x_second_half_reversed = torch.flip(x_second_half, dims=[0])

    #         low_x_first_half_reversed = F.avg_pool1d(x_first_half_reversed.transpose(0,2), kernel_size=3, stride=2)
    #         high_x_first_half_reversed = F.max_pool1d(x_first_half_reversed.transpose(0,2),kernel_size=3, stride=2)
    #         x_first_half_reversed = 0.9*F.interpolate(low_x_first_half_reversed, size=x_first_half.shape[0])+0.1*F.interpolate(high_x_first_half_reversed, size=x_first_half.shape[0])
    #         x_first_half_reversed = x_first_half_reversed.transpose(0,2)

    #         low_x_second_half_reversed = F.avg_pool1d(x_second_half_reversed.transpose(0,2), kernel_size=3, stride=2)
    #         high_x_second_half_reversed = F.max_pool1d(x_second_half_reversed.transpose(0,2),kernel_size=3, stride=2)
    #         x_second_half_reversed = 0.9*F.interpolate(low_x_second_half_reversed, size=x_second_half.shape[0])+0.1*F.interpolate(high_x_second_half_reversed, size=x_second_half.shape[0])
    #         x_second_half_reversed = x_second_half_reversed.transpose(0,2)

    #         # 相加反序后的两部分
    #         weight_first = weight_first.view(-1,1,1).to(x.device)
    #         weight_second = weight_second.view(-1,1,1).to(x.device)
    #         new_first_half = weight_first*x_first_half + weight_second* x_second_half_reversed
    #         new_second_half = weight_first*x_second_half + weight_second* x_first_half_reversed

    #         x=torch.cat([new_first_half,new_second_half],dim=0)

    #     else:
            
    #         num_frames=_x.shape[1]
            

    #         half_size = _x.shape[0] // 2
    #         x_first_half = x[ :half_size,:, :]
    #         x_second_half = x[ half_size:,:, :]

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[:,0,:], y_train=x_first_half[:,-1,:])
    #         # x_first_half[:,1:num_frames-2,:] = gp.predict(x_first_half[:,1:num_frames-2,:].transpose(1,0)).transpose(1,0).contiguous()

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[:,0,:], y_train=x_second_half[:,-1,:])
    #         # x_second_half[:,1:num_frames-2,:] = gp.predict(x_second_half[:,1:num_frames-2,:].transpose(1,0)).transpose(1,0).contiguous()
            
    #         gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[:,0,:], y_train=x_first_half[:,-1,:])
    #         mu_first,cov_first = gp.predict(x_first_half[:,1:num_frames-2,:].transpose(1,0))
    #         x_first_half[:,1:num_frames-2,:] = mu_first.transpose(1,0).contiguous()
    #         gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[:,0,:], y_train=x_second_half[:,-1,:])
    #         mu_second, cov_second = gp.predict(x_second_half[:,1:num_frames-2,:].transpose(1,0))
    #         x_second_half[:,1:num_frames-2,:] = mu_second.transpose(1,0).contiguous()
    #         #时序传播
    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[:,0,:], y_train=x_first_half[:,1,:])
    #         # x_first_half[:,1:num_frames-2,:] = gp.predict(x_first_half[:,0:num_frames-3,:].transpose(1,0)).transpose(1,0).contiguous()

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[:,0,:], y_train=x_second_half[:,1,:])
    #         # x_second_half[:,1:num_frames-2,:] = gp.predict(x_second_half[:,0:num_frames-3,:].transpose(1,0)).transpose(1,0).contiguous()

    #         weight_first = torch.sqrt(cov_first.diagonal(dim1=-2, dim2=-1)).mean(-1)
    #         weight_second = torch.sqrt(cov_second.diagonal(dim1=-2, dim2=-1)).mean(-1)
              

    #         # 反序操作
    #         x_first_half_reversed = torch.flip(x_first_half, dims=[1])
    #         x_second_half_reversed = torch.flip(x_second_half, dims=[1])

    #         low_x_first_half_reversed = F.avg_pool1d(x_first_half_reversed.transpose(1,2), kernel_size=3, stride=2)
    #         high_x_first_half_reversed = F.max_pool1d(x_first_half_reversed.transpose(1,2),kernel_size=3, stride=2)
    #         x_first_half_reversed = 0.9*F.interpolate(low_x_first_half_reversed, size=x_first_half.shape[1])+0.1*F.interpolate(high_x_first_half_reversed, size=x_first_half.shape[1])
    #         x_first_half_reversed = x_first_half_reversed.transpose(1,2)

    #         low_x_second_half_reversed = F.avg_pool1d(x_second_half_reversed.transpose(1,2), kernel_size=3, stride=2)
    #         high_x_second_half_reversed = F.max_pool1d(x_second_half_reversed.transpose(1,2),kernel_size=3, stride=2)
    #         x_second_half_reversed = 0.9*F.interpolate(low_x_second_half_reversed, size=x_second_half.shape[1])+0.1*F.interpolate(high_x_second_half_reversed, size=x_second_half.shape[1])
    #         x_second_half_reversed = x_second_half_reversed.transpose(1,2)

    #         weight_first = weight_first.view(1,-1,1).to(x.device)
    #         weight_second = weight_second.view(1,-1,1).to(x.device)
    #         new_first_half = weight_first*x_first_half + weight_second* x_second_half_reversed
    #         new_second_half = weight_first*x_second_half + weight_second* x_first_half_reversed
    #         x=torch.cat([new_first_half,new_second_half],dim=0)


    #     x = _x + x

    #     _x = self.attn2(self.norm2(x), context=context, mask=mask) + x

    #     if _x.shape[0]<_x.shape[1]:
    #         # num_frames=_x.shape[0]


    #         # half_size = _x.shape[0] // 2
    #         # x_first_half = x[:half_size]
    #         # x_second_half = x[half_size:]



    #         # num_frames=_x.shape[0]


    #         half_size = _x.shape[0] // 2
    #         num_frames=half_size
    #         x_first_half = x[:half_size]
    #         x_second_half = x[half_size:]

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[0,:,:], y_train=x_first_half[-1,:,:])
    #         # x_first_half[1:num_frames-2,:,:] = gp.predict(x_first_half[1:num_frames-2,:,:])

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[0,:,:], y_train=x_second_half[-1,:,:])
    #         # x_second_half[1:num_frames-2,:,:] = gp.predict(x_second_half[1:num_frames-2,:,:])

    #         gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[0,:,:], y_train=x_first_half[-1,:,:])
    #         x_first_half[1:num_frames-2,:,:], cov_first = gp.predict(x_first_half[1:num_frames-2,:,:])

    #         gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[0,:,:], y_train=x_second_half[-1,:,:])
    #         x_second_half[1:num_frames-2,:,:], cov_second = gp.predict(x_second_half[1:num_frames-2,:,:])

    #         #时序传播
    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[0,:,:], y_train=x_first_half[1,:,:])
    #         # # temp_x1 = x_first_half[0:num_frames-3,:,:]
    #         # x_first_half[1:num_frames-2,:,:] = gp.predict(x_first_half[0:num_frames-3,:,:])

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[0,:,:], y_train=x_second_half[1,:,:])
    #         # x_second_half[1:num_frames-2,:,:] = gp.predict(x_second_half[0:num_frames-3,:,:])

    #         weight_first = torch.sqrt(cov_first.diagonal(dim1=-2, dim2=-1)).mean(-1)
    #         weight_second = torch.sqrt(cov_second.diagonal(dim1=-2, dim2=-1)).mean(-1)

    #         # 反序操作
    #         x_first_half_reversed = torch.flip(x_first_half, dims=[0])
    #         x_second_half_reversed = torch.flip(x_second_half, dims=[0])

    #         low_x_first_half_reversed = F.avg_pool1d(x_first_half_reversed.transpose(0,2), kernel_size=3, stride=2)
    #         high_x_first_half_reversed = F.max_pool1d(x_first_half_reversed.transpose(0,2),kernel_size=3, stride=2)
    #         x_first_half_reversed = 0.9*F.interpolate(low_x_first_half_reversed, size=x_first_half.shape[0])+0.1*F.interpolate(high_x_first_half_reversed, size=x_first_half.shape[0])
    #         x_first_half_reversed = x_first_half_reversed.transpose(0,2)

    #         low_x_second_half_reversed = F.avg_pool1d(x_second_half_reversed.transpose(0,2), kernel_size=3, stride=2)
    #         high_x_second_half_reversed = F.max_pool1d(x_second_half_reversed.transpose(0,2),kernel_size=3, stride=2)
    #         x_second_half_reversed = 0.9*F.interpolate(low_x_second_half_reversed, size=x_second_half.shape[0])+0.1*F.interpolate(high_x_second_half_reversed, size=x_second_half.shape[0])
    #         x_second_half_reversed = x_second_half_reversed.transpose(0,2)

    #         # 相加反序后的两部分
    #         weight_first = weight_first.view(-1,1,1).to(x.device)
    #         weight_second = weight_second.view(-1,1,1).to(x.device)
    #         new_first_half = weight_first*x_first_half + weight_second* x_second_half_reversed
    #         new_second_half = weight_first*x_second_half + weight_second* x_first_half_reversed

    #         x=torch.cat([new_first_half,new_second_half],dim=0)

    #     else:
            
    #         num_frames=_x.shape[1]
            

    #         half_size = _x.shape[0] // 2
    #         x_first_half = x[ :half_size,:, :]
    #         x_second_half = x[ half_size:,:, :]

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[:,0,:], y_train=x_first_half[:,-1,:])
    #         # x_first_half[:,1:num_frames-2,:] = gp.predict(x_first_half[:,1:num_frames-2,:].transpose(1,0)).transpose(1,0).contiguous()

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[:,0,:], y_train=x_second_half[:,-1,:])
    #         # x_second_half[:,1:num_frames-2,:] = gp.predict(x_second_half[:,1:num_frames-2,:].transpose(1,0)).transpose(1,0).contiguous()
            
    #         gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[:,0,:], y_train=x_first_half[:,-1,:])
    #         mu_first,cov_first = gp.predict(x_first_half[:,1:num_frames-2,:].transpose(1,0))
    #         x_first_half[:,1:num_frames-2,:] = mu_first.transpose(1,0).contiguous()
    #         gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[:,0,:], y_train=x_second_half[:,-1,:])
    #         mu_second, cov_second = gp.predict(x_second_half[:,1:num_frames-2,:].transpose(1,0))
    #         x_second_half[:,1:num_frames-2,:] = mu_second.transpose(1,0).contiguous()
    #         #时序传播
    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_first_half[:,0,:], y_train=x_first_half[:,1,:])
    #         # x_first_half[:,1:num_frames-2,:] = gp.predict(x_first_half[:,0:num_frames-3,:].transpose(1,0)).transpose(1,0).contiguous()

    #         # gp = GaussianProcess(kernel=rbf_kernel, X_train=x_second_half[:,0,:], y_train=x_second_half[:,1,:])
    #         # x_second_half[:,1:num_frames-2,:] = gp.predict(x_second_half[:,0:num_frames-3,:].transpose(1,0)).transpose(1,0).contiguous()

    #         weight_first = torch.sqrt(cov_first.diagonal(dim1=-2, dim2=-1)).mean(-1)
    #         weight_second = torch.sqrt(cov_second.diagonal(dim1=-2, dim2=-1)).mean(-1)
              

    #         # 反序操作
    #         x_first_half_reversed = torch.flip(x_first_half, dims=[1])
    #         x_second_half_reversed = torch.flip(x_second_half, dims=[1])

    #         low_x_first_half_reversed = F.avg_pool1d(x_first_half_reversed.transpose(1,2), kernel_size=3, stride=2)
    #         high_x_first_half_reversed = F.max_pool1d(x_first_half_reversed.transpose(1,2),kernel_size=3, stride=2)
    #         x_first_half_reversed = 0.9*F.interpolate(low_x_first_half_reversed, size=x_first_half.shape[1])+0.1*F.interpolate(high_x_first_half_reversed, size=x_first_half.shape[1])
    #         x_first_half_reversed = x_first_half_reversed.transpose(1,2)

    #         low_x_second_half_reversed = F.avg_pool1d(x_second_half_reversed.transpose(1,2), kernel_size=3, stride=2)
    #         high_x_second_half_reversed = F.max_pool1d(x_second_half_reversed.transpose(1,2),kernel_size=3, stride=2)
    #         x_second_half_reversed = 0.9*F.interpolate(low_x_second_half_reversed, size=x_second_half.shape[1])+0.1*F.interpolate(high_x_second_half_reversed, size=x_second_half.shape[1])
    #         x_second_half_reversed = x_second_half_reversed.transpose(1,2)

    #         weight_first = weight_first.view(1,-1,1).to(x.device)
    #         weight_second = weight_second.view(1,-1,1).to(x.device)
    #         new_first_half = weight_first*x_first_half + weight_second* x_second_half_reversed
    #         new_second_half = weight_first*x_second_half + weight_second* x_first_half_reversed
    #         x=torch.cat([new_first_half,new_second_half],dim=0)
            
    #     x = _x + x


    #     x = self.ff(self.norm3(x)) + x


    #     return x
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=False, video_length=None,
                 image_cross_attention=False, image_cross_attention_scale_learnable=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                attention_cls=attention_cls,
                video_length=video_length,
                image_cross_attention=image_cross_attention,
                image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear


    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
    
    
class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, use_linear=False, only_self_att=True, causal_attention=False, causal_block_size=1,
                 relative_position=False, temporal_length=None):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert(temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
        if self.causal_attention:
            assert(temporal_length is not None)
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        temp_mask = None
        if self.causal_attention:
            # slice the from mask map
            temp_mask = self.mask[:,:t,:t].to(x.device)

        if temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
        else:
            mask = None

        if self.only_self_att:
            ## note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask)
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = repeat(
                        context[j],
                        't l con -> (t r) l con', r=(h * w) // t, t=t).contiguous()
                    ## note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j)
        
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in
    

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
