import tensorflow as tf
import numpy as np

def create_padding_mask(seq):
    """
    Indicates where values in the input are 0, and builds a mask 
    to be applied in the Attention layer.

    Since dimensions of the input to the attention layer is 
    (batch, head, time, features), we need to add dimensions to
    distribute the mask on the right axis position.

    The mask itself is applied to a shape (batch, heads, time, time)
    within the Attention layer.
    The last axis is where we want to apply the mask, since
    that is the axis the softmax is applied to. We are using the 
    softmax to do the actual masking.



    seq : Tensor of ints, shape (batch, time)
    Input sequence to the model (before applying embedding layer).

    returns: Mask 0,1 valued tensor of shape (batch, 1, 1, time). 
    The axis of value 1 is distributed across all axes in that dimension.
    """

    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(seq):
    """
    Creates a look ahead mask to be applied in the Attention layer.
    Mask is an upper triangular matrix.

    seq : Tensor of ints, shape (batch, time)
    Input sequence to the model (before applying embedding layer).

    returns : Tensor of 0/1 values, shape (time, time)
    """
    seq_len = seq.shape[1]
    return 1-tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


def create_combined_mask(seq):
    """
    Creates a mask consisting of both padding mask and look ahead mask

    seq : Tensor of ints, shape (batch, time)
    Input sequence to the model (before applying embedding layer).

    returns: Mask 0/1 valued tensor of shape (batch, 1, time, time). 
    """
    return tf.maximum(create_padding_mask(seq), create_look_ahead_mask(seq))

def combine_masks(mask1, mask2):
    return tf.maximum(mask1, mask2)



class Attention(tf.keras.layers.Layer):
    def __init__(self):
        """
        Dot product attention as in "Attention is all you need".

        """
        super(Attention, self).__init__()

    def call(self, q, k, v, mask=None):
        """
        q : query vector of shape (...,time,features)
        k : key vector of shape (...,time,features)
        v : value vector of shape (...,time,features)

        mask : Tensor of indicators for masked positions

        """
        prod = tf.matmul(q,k, transpose_b=True) # shape: (..., seq_len_q, seq_len_k)
        # scale
        prod = prod / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        
        # Add large negative values to masked positions before softmax in order to squash to 0.
        if mask is not None:
            prod += (mask * -1e9)

        prod = tf.keras.activations.softmax(prod) # shape: (..., seq_len_q, seq_len_k)
        return tf.matmul(prod, v) # shape: (..., seq_len_q, features_v)



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        """
        Mutlihead self attention from paper "Attention is all you need".

        dim : int
            size of feature/embedding dimension

        heads : int
            number of splits to split the feature/embedding dimension.

        """
        super(MultiHeadAttention, self).__init__()
        
        assert  dim % heads == 0 , "dim % heads == 0"

        self.dim = dim
        self.heads = heads

        self.wQ = tf.keras.layers.Dense(dim)
        self.wK = tf.keras.layers.Dense(dim)
        self.wV = tf.keras.layers.Dense(dim)
        self.linear = tf.keras.layers.Dense(dim)
    
    def call(self, q, k, v, mask = None):
        """
        x : tensor of shape (batch, time, dim)

        returns: tensor of shape (batch, time, dim)

        """
        batch_size = q.shape[0]

        Q = self.wQ(q)
        K = self.wK(k)
        V = self.wV(v)
        
        def transform(w):
            """
            Splits up feature dimension into multiple dimensions.
            This lets us apply the attention mechanism to all heads simultanously

            result shape: (batch, heads, time, dim//heads)

            """
            reshaped = tf.reshape(w, [batch_size, -1, self.heads, self.dim // self.heads])
            transposed = tf.transpose(reshaped, [0,2,1,3]) # Switch time and heads dimension
            return transposed

        def reverse_transform(w):
            """
            Reverses previous transform and combines the split dimension:
            
            result shape: (batch, time, features)

            """
            transposed = tf.transpose(w, [0,2,1,3]) # Switch time and heads dimension
            reshaped = tf.reshape(transposed, [batch_size, -1, self.dim])
            return reshaped

        # Splits up the full matrix into muliple lower dimension matrices
        Q = transform(Q)
        K = transform(K)
        V = transform(V)

        result = Attention()(Q, K, V, mask) # (batch, heads, time, features//heads)

        # Combines the matrices again
        result = reverse_transform(result) # (batch, time, features)
        return self.linear(result)


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(FeedForwardNetwork, self).__init__()

        self.dim = dim

        self.inner_layer = tf.keras.layers.Dense(dim * 4, activation = 'relu')
        self.outer_layer = tf.keras.layers.Dense(dim)

    def call(self, x):
        out = self.inner_layer(x)
        out = self.outer_layer(out)
        return out

class AddAndNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(AddAndNorm, self).__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, y):
        return self.norm(x + y)
    

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super(EncoderBlock, self).__init__()

        self.dim = dim
        self.heads = heads

        self.attention = MultiHeadAttention(dim, heads)
        self.ffnetwork = FeedForwardNetwork(dim)

        self.add_norm_1 = AddAndNorm()
        self.add_norm_2 = AddAndNorm()

        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, x, training = False, mask = None):
        # Self attention
        a  = self.attention(x, x, x, mask = mask)
        a = self.dropout1(a, training = training)
        a = self.add_norm_1(x, a)
        
        b = self.ffnetwork(a)
        b = self.dropout1(b, training = training)
        b = self.add_norm_2(a, b)

        return b

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, dim, heads):
        super(Encoder, self).__init__()
                
        self.num_layers = num_layers
        self.dim = dim
        self.heads = heads

        self.encoder_blocks = [EncoderBlock(dim, heads) for i in range(num_layers)]

    def call(self, x, training = False, mask = None):
        """
        x : tensor
            shape (batch, time, features) - Apply embedding transform prior to this.
        """
        assert len(x.shape) == 3, "assert len(x.shape) == 3"

        for block in self.encoder_blocks:
            x = block(x, training = training, mask = mask)
        
        return x

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super(DecoderBlock, self).__init__()

        self.dim = dim
        self.heads = heads

        self.attention_1 = MultiHeadAttention(dim, heads)
        self.attention_2 = MultiHeadAttention(dim, heads)
        self.ffnetwork = FeedForwardNetwork(dim)

        self.add_norm_1 = AddAndNorm()
        self.add_norm_2 = AddAndNorm()
        self.add_norm_3 = AddAndNorm()

        self.dropout_1 = tf.keras.layers.Dropout(0.1)
        self.dropout_2 = tf.keras.layers.Dropout(0.1)
        self.dropout_3 = tf.keras.layers.Dropout(0.1)

    def call(self, x, encoder_output, training = False, first_layer_mask = None, second_layer_mask = None):

        """
        x : tensor
            shape (batch, time, features) - Apply embedding transform prior to this.

        encoder_output : tensor
            output from the encoder step, to be passed in to the decoder.

        training : bool
            True if training, else False.

        first_layer_mask : tensor
            mask provided to the first multi head attention encountered in the decoder

        second_layer_mask : tensor
            mask provided to the second multi head attention encountered in the decoder
        """

        a = self.attention_1(x, x, x, mask = first_layer_mask)
        a = self.dropout_1(a, training = training)
        a = self.add_norm_1(x, a)

        b = self.attention_2(q = a, k = encoder_output, v = encoder_output, mask = second_layer_mask)
        b = self.dropout_2(b, training = training)
        b = self.add_norm_2(a, b)

        c = self.ffnetwork(b)
        c = self.dropout_3(c, training = training)
        c = self.add_norm_3(b, c)

        return c



class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, dim, heads):
        super(Decoder, self).__init__()
                
        self.num_layers = num_layers
        self.dim = dim
        self.heads = heads

        self.decoder_blocks = [DecoderBlock(dim, heads) for i in range(num_layers)]

    def call(self, x, encoder_output, training = False, first_layer_mask = None, second_layer_mask = None):
        """
        x : tensor
            shape (batch, time, features) - Apply embedding transform prior to this.
        """
        assert len(x.shape) == 3, "assert len(x.shape) == 3"

        for block in self.decoder_blocks:
            x = block(x, encoder_output, training = training, 
                        first_layer_mask = first_layer_mask, 
                        second_layer_mask = second_layer_mask)
        
        return x



class Transformer(tf.keras.Model):
    def __init__(self, num_layers, dim, heads, vocab_size):
        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.dim = dim
        self.heads = heads
        self.vocab_size = vocab_size

        self.encoder = Encoder(num_layers, dim, heads)
        self.decoder = Decoder(num_layers, dim, heads)
        self.linear = tf.keras.layers.Dense(vocab_size)

        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = dim)
        #TODO: positional encoding

    def call(self, x, y, training = False):
        padding_mask_x = create_padding_mask(x)
        padding_mask_y = create_padding_mask(y)
        combined_mask_y = create_combined_mask(y)
        
        x = self.embedding(x)
        y = self.embedding(y)

        encoder_output = self.encoder(x = x, training = training, mask = padding_mask_x)
        decoder_output = self.decoder(x=y, encoder_output = encoder_output, training = training, 
                                        first_layer_mask = combined_mask_y, second_layer_mask = padding_mask_y)

        # Logits
        output = self.linear(decoder_output)
        return output

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, dim, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.dim = dim
    self.dim = tf.cast(self.dim, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.dim) * tf.math.minimum(arg1, arg2)
