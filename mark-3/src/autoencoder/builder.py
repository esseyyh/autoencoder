import torch.nn as nn
import torch
from torchsummary import summary 
import math
import torch.nn.functional as F

class resnet(nn.Module):
    """ResNet block """
   
    def __init__(self,
                 fan_in : int,
                 fan_out: int,
                 resnet_groups:int =32
                 ):
        super().__init__()
        self.layer1=nn.Conv2d(fan_in, fan_out,kernel_size=3, stride=1, padding=1)
        self.gn1=nn.GroupNorm(resnet_groups,fan_in)
        self.layer2=nn.Conv2d(fan_out,fan_out,kernel_size=3,stride=1,padding= 1) 
        self.gn2=nn.GroupNorm(resnet_groups,fan_out)
        self.act=nn.SiLU()
        
        if fan_in==fan_out:
            self.proj_layer=nn.Identity()
        else:
            self.proj_layer = nn.Conv2d(fan_in,fan_out, kernel_size=1, stride=1, padding=0)

            
        
    def forward(self,x):    
    
        out=self.layer1(self.act(self.gn1(x)))
        out=self.layer2(self.act(self.gn2(out)))
        

        
        return (out +  self.proj_layer(x) )
            

        
            

class block(nn.Module):
    """ a stack of resnet blocks for VAE """
    def __init__(self,
                 fan_in:int,    
                 fan_out:int,
                 resnetlayer :int,
                 resnet_groups:int =32,
                 down:bool = False,
                 up:bool=False):

        super().__init__()
        self.down=down
        self.up=up
        

        self.res_layers = nn.ModuleList([resnet(fan_in,
                       fan_out,resnet_groups=resnet_groups) 
                       for i in range (resnetlayer)
                       ])
        if down :
            self.down_step = nn.Conv2d(fan_out,fan_out,3,2,0)
        if up:
            self.up_step_c= nn.Conv2d(fan_out,fan_out,3,1,1)
            self.up_step = nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self,x):

        

        for res in self.res_layers:
            x=res(x)
          
        

        if self.down:
            x=F.pad(x,(0,1,0,1))
            x=self.down_step(x)
        if self.up:
            x=self.up_step(self.up_step_c(x))
        return x



class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
       
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
       
        input_shape = x.shape 
        
        batch_size, sequence_length, d_embed = input_shape 

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head) 

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2) 

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output




class Attentionblock(nn.Module):
    def __init__(self, channels):
        super().__init__()


        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        

        residue = x 

        
        x = self.groupnorm(x)

        B, C, H ,W = x.shape
        
       
        x = x.view((B, C, H * W))
        
        
        x = x.transpose(-1, -2)
        
        
        x = self.attention(x)
        
        
        x = x.transpose(-1, -2)
        
        
        x = x.view((B, C, H, W))
        
        
        x += residue

        return x 
    
        
        