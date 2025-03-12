import torch.nn as nn
import warnings
import sys
import os
import torch
from sklearn.metrics import auc, roc_curve
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
import warnings
warnings.simplefilter("ignore")
import numpy as np
##########################

class MultiHead_Self_Attention(nn.Module):
    def __init__(self,
                 embed_dim0,
                 embed_dim, 
                 num_heads,
                 masked=True):
        super(MultiHead_Self_Attention, self).__init__()
        '''
        MultiHead self attention with interaction matrix U. 
        The diemsnion of U is (batch,num of heads, particle tokens, particle tokens)
        
        Args:
            embed_dim0: int, dim of the feature tokens
            embed_dim: int,hidden dimension, "scaled attention"
            num_heads: int,number of attention heads
            masked: polean, using of the attention mask to remove the padded points
            
        return:
              1- output of attention heads, with dim (batch,particle tokens, feature tokens)
              2- attention weights, with dim (batch, particle tokens, particle tokens)
        '''
        self.masked = masked
        self.embed_dim0 = embed_dim0
        if embed_dim % num_heads == 0:
    
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim =  embed_dim // num_heads
        else:    
            self.embed_dim = embed_dim + (num_heads - embed_dim % num_heads)
            self.num_heads = num_heads
            self.head_dim =  self.embed_dim // num_heads
            
        # Initialize the linear layers
        self.q_linear = nn.Linear(self.embed_dim0, self.embed_dim, bias=False)
        self.k_linear = nn.Linear(self.embed_dim0, self.embed_dim, bias=False)
        self.v_linear = nn.Linear(self.embed_dim0, self.embed_dim, bias=False)
        self.out_linear = nn.Linear(self.embed_dim, self.embed_dim0, bias=False)
    
    def att_mask(self, input_):
        '''
         Function to create attention mask with 1 for unpadded points and 0 for padded points
    
         arg1: input data set with dim (batch_size, n_particles,n_features)
         output: mask tensor of dim (batch_size, number of heads, n_particles,n_particles)
        '''

        mask = (input_.sum(dim=-1) != 0).float()
        mask = mask[:, :, None,:] 
        mask = mask.repeat(1,1, mask.size(-1),1)
    
        return mask
            
         
    def scaled_dot_product_attention(self, Q, K, V):
        """
        Computes scaled dot-product attention.
        dim of U: batch_size x num_heads x particle_tokens x particle_tokens
        """
        d_k = Q.size(-1)

        scores = (torch.matmul(Q, K.transpose(-2, -1))) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
       
        if self.masked:
            mask = self.att_mask(Q)    
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, query):
        batch_size = query.size(0)
        
        Q = self.q_linear(query)
        K = self.k_linear(query)
        V = self.v_linear(query)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Apply final linear layer # W matrix
        output = self.out_linear(attn_output)

        return output


###############################################
class TransformerLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 h_dim=500, 
                 expansion_factor=4,
                 n_heads=10,
                 masked=True,):
        super(TransformerLayer, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        '''
        To be done: 
               Here, the hidden dimension is fixed, may be we need to adopt it in the future.
        '''

        self.input_dim = input_dim
        self.n_heads= n_heads
        self.masked = masked
        self.expansion_factor = expansion_factor
        self.h_dim = h_dim
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.attention = MultiHead_Self_Attention(self.input_dim,self.h_dim, self.n_heads,self.masked)
        self.norm1 = nn.RMSNorm(self.input_dim)
        self.norm2 = nn.RMSNorm(self.input_dim)
        self.feed_forward = nn.Sequential(
                          nn.RMSNorm(self.input_dim),
                          nn.Linear(self.input_dim, self.expansion_factor*self.input_dim),
                          nn.SiLU(),
                          nn.RMSNorm(self.expansion_factor*self.input_dim), 
                          nn.Linear(self.expansion_factor*self.input_dim, self.input_dim))
    def forward(self,query):

        attention_out= self.attention(self.norm1(query))  
        attention_residual_out = self.norm2(attention_out) + query  
        norm1_out = self.dropout1(attention_residual_out)
        feed_fwd_out = self.feed_forward(norm1_out) 
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(feed_fwd_residual_out)

        return norm2_out
###########################################
###########################################
###########################################
###########################################
class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self,  
        input_dim,
        embed_dim=[128,64], 
        h_dim=200,
        num_layers=2, 
        expansion_factor=4,
        n_heads=10,
        masked=True):
        super(TransformerEncoder, self).__init__()
       

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.masked = masked
        self.expansion_factor = expansion_factor
        self.n_heads=n_heads
        self.num_layers = num_layers
        self.embed = Embedding(self.input_dim,self.embed_dim)
        self.layers = nn.ModuleList([TransformerLayer(self.embed_dim[-1],self.h_dim,self.expansion_factor,self.n_heads,self.masked) for i in range(self.num_layers)])
        self.n1 = nn.LayerNorm(self.embed_dim[-1])
        self.n2 = nn.Linear(9,1)
        
    def forward(self, x):
        x_new = self.embed(x)

        out = self.n1(x_new)


        for layer in self.layers:
            out = layer(out)


        Transformer_output = out.mean(dim=2)
        
        output = self.n2(Transformer_output)   
        return output  
        

############################
############################

########################################################
#########################
class Embedding(nn.Module):
    """
    Embedding of the feature tokens in both, Transformer and EEDGNN.

    args:
        input_dim: int, dim of the feature tokens
        layers_dim: list, list of the number of neurons of each FC layer

    return:
          MLP network.
    """

    def __init__(self, input_dim, layers_dim):
        super(Embedding, self).__init__()

        self.layers_dim = layers_dim
        self.layers = nn.ModuleList()

        for i in layers_dim[1:]:
            self.layers.append(nn.LayerNorm(input_dim))
            self.layers.append(nn.Linear(input_dim, i))
            self.layers.append(nn.GELU())
            input_dim = i

    def forward(self, input_):
        x = input_
        for layer in self.layers:
            x = layer(x)
        return x


###########################
###########################
###########################
class mlp(nn.Module):
    """
    Final mlp of the network, after pooling the Transformer encoder output.

    args:
        input_dim: int, dim of the feature tokens of Transformer encoder output
        layers_dim: list, list of the number of neurons of each FC layer

    return:
          MLP network with one neuron output and Sigmoid activation
    """

    def __init__(self, input_dim, layers_dim):
        super(mlp, self).__init__()
        self.layers_dim = layers_dim
        self.layers = nn.ModuleList()

        for i in layers_dim[1:]:
            self.layers.append(nn.LayerNorm(input_dim))
            self.layers.append(nn.Linear(input_dim, i))
            self.layers.append(nn.GELU())
            input_dim = i
        self.layers.append(nn.Linear(i, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, input_):
        x = input_
        for layer in self.layers:
            x = layer(x)
        return x



    
    def forward(self,inp_T):
        ''' 
        input_T: dim (batch, particle tokens, feature tokens)
        input_E: dim (batch, particle tokens, particle tokens, pairwise features)
        '''
        
                
        Transformer_out = self.encoder(inp_T)
        if self.pooling == 'avg':
            Transformer_output = Transformer_out.mean(dim=2)

        elif self.pooling == 'max':
            Transformer_output = Transformer_out.max(dim=2)[0]

        elif self.pooling == 'sum':
            Transformer_output = Transformer_out.sum(dim=2)

        output = self.mlp(Transformer_output)   
      
        return  output

