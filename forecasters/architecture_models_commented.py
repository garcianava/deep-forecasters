import numpy as np
import tensorflow.compat.v1 as tf

# required for TFA MultiHeadAttention
import typing
import warnings


class MultiHeadAttention(tf.keras.layers.Layer):
    r"""MultiHead Attention layer.
    Defines the MultiHead Attention operation as described in
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762) which takes
    in the tensors `query`, `key`, and `value`, and returns the dot-product attention
    between them:
    >>> mha = MultiHeadAttention(head_size=128, num_heads=12)
    >>> query = np.random.rand(3, 5, 4) # (batch_size, query_elements, query_depth)
    >>> key = np.random.rand(3, 6, 5) # (batch_size, key_elements, key_depth)
    >>> value = np.random.rand(3, 6, 6) # (batch_size, key_elements, value_depth)
    >>> attention = mha([query, key, value]) # (batch_size, query_elements, value_depth)
    >>> attention.shape
    TensorShape([3, 5, 6])
    If `value` is not given then internally `value = key` will be used:
    >>> mha = MultiHeadAttention(head_size=128, num_heads=12)
    >>> query = np.random.rand(3, 5, 5) # (batch_size, query_elements, query_depth)
    >>> key = np.random.rand(3, 6, 10) # (batch_size, key_elements, key_depth)
    >>> attention = mha([query, key]) # (batch_size, query_elements, key_depth)
    >>> attention.shape
    TensorShape([3, 5, 10])
    Args:
        head_size: int, dimensionality of the `query`, `key` and `value` tensors
            after the linear transformation.
        num_heads: int, number of attention heads.
        output_size: int, dimensionality of the output space, if `None` then the
            input dimension of `value` or `key` will be used,
            default `None`.
        dropout: float, `rate` parameter for the dropout layer that is
            applied to attention after softmax,
        default `0`.
        use_projection_bias: bool, whether to use a bias term after the linear
            output projection.
        return_attn_coef: bool, if `True`, return the attention coefficients as
            an additional output argument.
        kernel_initializer: initializer, initializer for the kernel weights.
        kernel_regularizer: regularizer, regularizer for the kernel weights.
        kernel_constraint: constraint, constraint for the kernel weights.
        bias_initializer: initializer, initializer for the bias weights.
        bias_regularizer: regularizer, regularizer for the bias weights.
        bias_constraint: constraint, constraint for the bias weights.
    Call Args:
        inputs:  List of `[query, key, value]` where
            * `query`: Tensor of shape `(..., query_elements, query_depth)`
            * `key`: `Tensor of shape '(..., key_elements, key_depth)`
            * `value`: Tensor of shape `(..., key_elements, value_depth)`, optional, if not given `key` will be used.
        mask: a binary Tensor of shape `[batch_size?, num_heads?, query_elements, key_elements]`
        which specifies which query elements can attendo to which key elements,
        `1` indicates attention and `0` indicates no attention.
    Output shape:
        * `(..., query_elements, output_size)` if `output_size` is given, else
        * `(..., query_elements, value_depth)` if `value` is given, else
        * `(..., query_elements, key_depth)`
    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        output_size: int = None,
        dropout: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
        kernel_regularizer: typing.Union[str, typing.Callable] = None,
        kernel_constraint: typing.Union[str, typing.Callable] = None,
        bias_initializer: typing.Union[str, typing.Callable] = "zeros",
        bias_regularizer: typing.Union[str, typing.Callable] = None,
        bias_constraint: typing.Union[str, typing.Callable] = None,
        **kwargs
    ):
        warnings.warn(
            "`MultiHeadAttention` will be deprecated in Addons 0.13. "
            "Please use `tf.keras.layers.MultiHeadAttention` instead.",
            DeprecationWarning,
        )

        super().__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self._dropout_rate = dropout

    def build(self, input_shape):

        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Linear transformations
        query = tf.einsum("...NI , HIO -> ...NHO", query, self.query_kernel)
        key = tf.einsum("...MI , HIO -> ...MHO", key, self.key_kernel)
        value = tf.einsum("...MI , HIO -> ...MHO", value, self.value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=query.dtype)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._dropout_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
            kernel_initializer=tf.keras.initializers.serialize(self.kernel_initializer),
            kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer),
            kernel_constraint=tf.keras.constraints.serialize(self.kernel_constraint),
            bias_initializer=tf.keras.initializers.serialize(self.bias_initializer),
            bias_regularizer=tf.keras.regularizers.serialize(self.bias_regularizer),
            bias_constraint=tf.keras.constraints.serialize(self.bias_constraint),
        )

        return config


# build a mask for self-attention on the autoregressive transformer decoder
def get_decoder_mask(self_attention_inputs):
    # self_attention_input shape is (?, n_timesteps, n_features)
    # get the dimension value of n_timesteps and build the mask
    n_timesteps = self_attention_inputs.shape[1]
    mask = tf.convert_to_tensor(np.tril(np.ones([n_timesteps, n_timesteps]), 0),
                                dtype=tf.float32)
    return mask


# base transformer encoder layer from # https://keras.io/examples/nlp/text_classification_with_transformer/
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = MultiHeadAttention(head_size=embed_dim, num_heads=num_heads)
        self.ff_layer = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"),
             tf.keras.layers.Dense(embed_dim)]
        )
        self.add_norm_layer_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add_norm_layer_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs):
        attention_output = self.attention_layer([inputs, inputs])
        attention_output = self.dropout_1(attention_output)
        input_to_ffn = self.add_norm_layer_1(inputs + attention_output)
        ffn_output = self.ff_layer(input_to_ffn)
        ffn_output = self.dropout_2(ffn_output)
        return self.add_norm_layer_2(input_to_ffn + ffn_output)


# base transformer encoder layer from # https://keras.io/examples/nlp/text_classification_with_transformer/
# modified to include masked self attention for autoregressive transformer decoder
# ToDo: get the number of timesteps from the input shape
# in the meantime, pass this value as an argument for the encoder layer
class ARDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(ARDecoderLayer, self).__init__()
        # multi-head attention initialization
        self.attention_layer = MultiHeadAttention(head_size=embed_dim, num_heads=num_heads)
        self.ff_layer = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"),
             tf.keras.layers.Dense(embed_dim)]
        )
        self.add_norm_layer_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add_norm_layer_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask):
        # mask for self-attention is passed to MHA on call
        attention_output = self.attention_layer([inputs, inputs], mask=mask)
        attention_output = self.dropout_1(attention_output)
        input_to_ffn = self.add_norm_layer_1(inputs + attention_output)
        ffn_output = self.ff_layer(input_to_ffn)
        ffn_output = self.dropout_2(ffn_output)
        return self.add_norm_layer_2(input_to_ffn + ffn_output)


# Transformer-encoder
# hourly-resolution only
class TRFENCD(object):
    # pass features (set of tensors) as inputs at call operator
    def __call__(self, features, model_params):
        # supervised learning database as input for the transformer encoder

        num_timesteps = model_params['embedding']['hourly']

        # dimensionality of Q, K, V
        embed_dim = model_params['model_dimension']

        # a simple Conv1D layer to project time series data (scalar) to d_model
        # ToDo: pass convolutional parameters from configuration file
        value_embedding_layer = tf.keras.layers.Conv1D(filters=embed_dim,
                                                       kernel_size=3,
                                                       activation="relu",
                                                       padding="same")(features['hourly'])
        # value_embedding_layer is (?, num_timesteps, embed_dim)

        # ToDo: use positional encoding from Flow Forecast
        positions_to_encode = tf.range(start=0, limit=num_timesteps, delta=1)
        # positions_to_encode is a tensor from 0 to num_timesteps-1 (V.gr. 0-255)

        position_embedding_layer = tf.keras.layers.Embedding(input_dim=num_timesteps,
                                                             output_dim=embed_dim)(positions_to_encode)
        # position_embedding_layer is (1, num_timesteps, embed_dim)

        input_to_transformer_block = value_embedding_layer + position_embedding_layer
        # input_to_transformer_block is (?, num_timesteps, embed_dim)

        encoder = list()

        encoder.append(input_to_transformer_block)

        # iterate on encoder structure to build encoder levels
        for level in np.arange(len(model_params['encoder']['num_heads'])):
            encoder.append(
                EncoderLayer(embed_dim=embed_dim,
                             num_heads=model_params['encoder']['num_heads'][level],
                             ff_dim=model_params['encoder']['ff_dim'][level],
                             dropout=model_params['encoder']['dropout'][level])(encoder[-1])
            )

        # num_heads = model_params['num_heads']
        # ff_dim = model_params['ff_dim']
        # dropout = model_params['dropout']

        # encoder_layer_1 = EncoderLayer(embed_dim=embed_dim,
        #                                num_heads=num_heads,
        #                                ff_dim=ff_dim,
        #                                dropout=dropout)

        # encoder_layer_2 = EncoderLayer(embed_dim=embed_dim,
        #                                num_heads=num_heads,
        #                                ff_dim=ff_dim,
        #                                dropout=dropout)

        # output_from_encoder_1 = encoder_layer_1(input_to_transformer_block)
        # output_from_encoder_2 = encoder_layer_2(output_from_encoder_1)
        # output_from_encoder is (?, num_timesteps, embed_dim)

        # processing the output from transformer block towards the target
        # case 1: based on TransformerBlock example at
        # https://keras.io/examples/nlp/text_classification_with_transformer/

        output_from_pooling = tf.keras.layers.GlobalAveragePooling1D()(encoder[-1])
        # output_from_pooling is (?, embed_dim)

        no_targets = model_params['no_targets']

        repeated = tf.keras.layers.RepeatVector(no_targets)(output_from_pooling)
        # repeated is (?, no_targets, embed_dim)

        # build simple decoder layer as a list
        decoder = list()

        decoder.append(repeated)

        # iterate on decoder structure to build decoder levels
        for level in np.arange(len(model_params['decoder']['structure'])):
            block_name = 'decoder_{}'.format(level)
            decoder.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dropout(model_params['decoder']['dropout'][level])
                )(decoder[-1])
            )
            decoder.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(
                        units=model_params['decoder']['structure'][level],
                        activation=model_params['decoder']['activation'][level],
                        name=block_name)
                )(decoder[-1])
            )

        # first_dropout = tf.keras.layers.Dropout(0.1)

        # distributed_first_dropout = tf.keras.layers.TimeDistributed(first_dropout)(repeated)
        # # distributed_first_dropout is (?, no_targets, embed_dim)

        # units_in_first_dense = 32
        # first_dense = tf.keras.layers.Dense(units_in_first_dense, activation="relu")

        # distributed_first_dense = tf.keras.layers.TimeDistributed(first_dense)(distributed_first_dropout)
        # # distributed_first_dense is (?, no_targets, units_in_first_dense)

        # second_dropout = tf.keras.layers.Dropout(0.1)

        # distributed_second_dropout = tf.keras.layers.TimeDistributed(second_dropout)(distributed_first_dense)
        # # distributed_second_dropout is (?, no_targets, units_in_first_dense)

        # units_in_second_dense = 1
        # second_dense = tf.keras.layers.Dense(units_in_second_dense, activation="sigmoid")

        # distributed_second_dense = tf.keras.layers.TimeDistributed(second_dense)(distributed_second_dropout)
        # # distributed_second_dense is (?, no_targets, units_in_second_dense)

        forecast = decoder[-1]
        return forecast


# Autoregressive Transformer-decoder
# hourly-resolution only
class ARTRFDC(object):
    # pass features (source tensors: values and positional encodings) as inputs at call operator
    def __call__(self, features, model_params):
        # supervised learning database as input for the transformer encoder

        # the number of timesteps is the length of the input sequence,
        # (the embedding dimension from the SLDB)
        # ToDo: initially, get this value from the configuration dictionary,
        #   (later, it will have to be automatically obtained from the source/target tensors shape)
        num_timesteps = model_params['num_timesteps']

        # number of features is the active load value (main feature)
        # plus the six components of the sine-cosine pos encoding on the hour, day, and month
        # important: there is no value embedding, therefore d_model is very low
        d_model = model_params['model_dimension']
        # ToDo: use value embedding to a high-dimensional space and compare results
        # ToDo: use a different global positional encoding system and compare results

        # abstract the architecture model as in TRFENCD class
        # to pass all architecture parameters from configuration file

        # a list to concatenate all AR decoder layers
        ar_decoder = list()

        # first input to the AR decoder is the source feature
        ar_decoder.append(features['source'])

        # use the same mask for all the ARDecoderLayers in the ARDecoder
        mask = get_decoder_mask(features['source'])

        # iterate on the ARDecoder structure to build its layer levels
        # the number of layers can be obtained from num_heads, ff_dim, or dropout
        for level in np.arange(len(model_params['num_heads'])):
            ar_decoder.append(
                ARDecoderLayer(embed_dim=d_model,
                               num_heads=model_params['num_heads'][level],
                               ff_dim=model_params['ff_dim'][level],
                               dropout=model_params['dropout'][level])(ar_decoder[-1], mask)
            )

        # ToDo: get the all the dense layer parameters from the configuration file
        units_in_first_dense = 1
        first_dense = tf.keras.layers.Dense(units_in_first_dense, activation="sigmoid")

        # get the output of the AR Decoder from the last entry in the list
        distributed_first_dense = tf.keras.layers.TimeDistributed(first_dense)(ar_decoder[-1])
        # distributed_first_dense is (?, 168, 1)

        forecast = distributed_first_dense
        return forecast



class DMSLSTM(object):
    # pass features (set of tensors) as inputs at call operator
    def __call__(self, features, model_params):
        # supervised learning database as input for the LSTM layers

        # collect one-hot timestamps for week/day and day/hour to merge them as features
        # oh:wd: encoded day of the week
        flat_one_hot_week_day = tf.keras.layers.Flatten()(features['oh_wd'])
        # oh_dh: encoded hour of the day
        flat_one_hot_day_hour = tf.keras.layers.Flatten()(features['oh_dh'])
        one_hot_merge = tf.keras.layers.concatenate([flat_one_hot_week_day, flat_one_hot_day_hour])

        # a dictionary to store the resolution-based LSTM stacks
        lstm = dict()
        # a dictionary to store the dropout layers for resolution-based LSTM stacks
        dropout = dict()

        # build the LSTM stacks for three time resolutions
        for resolution in ['hourly', 'daily', 'weekly']:
            lstm[resolution] = list()
            lstm[resolution].append(features[resolution])
            for level in np.arange(len(model_params[resolution]['structure'])):
                is_this_last_level = level == len(model_params[resolution]['structure']) - 1
                block_name = 'lstm_{}_{}'.format(resolution, level)
                lstm[resolution].append(
                    tf.keras.layers.LSTM(
                        units=model_params[resolution]['structure'][level],
                        # if multi-level, then LSTM blocks between first
                        # and the one before last must return sequences
                        return_sequences=not is_this_last_level,
                        dropout=model_params[resolution]['dropout'],
                        unroll=model_params[resolution]['unroll'],
                        implementation=model_params[resolution]['implementation_mode'],
                        name=block_name)(lstm[resolution][-1])
                )
            # ToDo: verify what type of dropout regularization can be applied inside the LSTM block
            # point the dropout to the last LSTM cell on the stack
            dropout[resolution] = tf.keras.layers.Dropout(
                model_params[resolution]['dropout'])(lstm[resolution][-1])

        # merge results of DMSLSTM branches
        if model_params['use_timestamps']:
            lstm_merge = tf.keras.layers.concatenate([dropout['hourly'],
                                                      dropout['daily'],
                                                      dropout['weekly'],
                                                      one_hot_merge])
        else:
            lstm_merge = tf.keras.layers.concatenate([dropout['hourly'],
                                                      dropout['daily'],
                                                      dropout['weekly']])

        # build dense layer as a list
        dense_layer = list()
        # input to dense layer is the vector LSTM results + timestamps
        dense_layer.append(lstm_merge)
        # iterate on dense structure to build dense levels
        for level in np.arange(len(model_params['dense']['structure'])):
            block_name = 'dense_{}'.format(level)
            dense_layer.append(
                tf.keras.layers.Dense(
                    units=model_params['dense']['structure'][level],
                    activation=model_params['dense']['activation'][level],
                    name=block_name
                )(dense_layer[-1])
            )

        # finally, make the forecast equal to the last layer of 'Dense'
        # (which is referenced by the most recent record in the dense_layer list
        forecast = dense_layer[-1]
        # reshape the forecast tensor to be consistent with target in the parsed datasets,
        # target dimension for multi-step forecasting is equal
        # to the last value in params['dense']['structure']
        # (and this value remains in params['dense']['structure'][level]
        # after 'dense' layer construction)
        # however, let's use a direct reference, just to be sure
        no_targets = model_params['dense']['structure'][-1]
        forecast = tf.keras.layers.Reshape((no_targets, 1))(forecast)

        return forecast


# Stacked LSTM-based encoder - decoder with no attention mechanism
# hourly-resolution only
# based on Jeremy Wortz' code from
# https://stackoverflow.com/questions/50915634/multilayer-seq2seq-model-with-lstm-in-keras
class EDSLSTM(object):
    # pass features (set of tensors) as inputs at call operator
    def __call__(self, features, model_params):
        # supervised learning database as input for the LSTM layers
        # start with [256, 256] LSTM stacks for encoder and decoder, generalize later

        # ToDo: annotate all tensor dimensions to verify consistency
        encoder_stack_h0 = tf.keras.layers.LSTM(
            units=model_params['encoder']['no_hidden'][0],
            activation=model_params['encoder']['activation'][0],
            dropout=model_params['encoder']['dropout'][0],
            recurrent_dropout=model_params['encoder']['recurrent_dropout'][0],
            return_sequences=True,
            return_state=False,
            name='encoder_level_0')(features['hourly'])
        # features['hourly'] is a tensor of shape (?, 64, 1) (samples, timesteps, features)
        # return_sequences=True, then
        # encoder_stack_h has shape (?, features['hourly'].timesteps, no_hidden) (?, 64, 256)

        encoder_stack_h1 = tf.keras.layers.LSTM(
            units=model_params['encoder']['no_hidden'][1],
            activation=model_params['encoder']['activation'][1],
            dropout=model_params['encoder']['dropout'][1],
            recurrent_dropout=model_params['encoder']['recurrent_dropout'][1],
            # return sequences are only required for the attention mechanism
            return_sequences=True,
            return_state=False,
            name='encoder_level_1')(encoder_stack_h0)
        # return_sequences=True, therefore
        # encoder_stack_h has shape (?, 64, 256)

        encoder_stack_h2, encoder_last_h2, encoder_last_c2 = tf.keras.layers.LSTM(
            units=model_params['encoder']['no_hidden'][2],
            activation=model_params['encoder']['activation'][2],
            dropout=model_params['encoder']['dropout'][2],
            recurrent_dropout=model_params['encoder']['recurrent_dropout'][2],
            return_sequences=True,
            return_state=True,
            name='encoder_level_2')(encoder_stack_h1)
        # return_sequences=True, therefore
        # encoder_stack_h2 has shape (?, encoder_stack_h1.timesteps, no_hidden) (?, 64, 256)
        # encoder_last_h2 has shape (?, no_hidden) (?, 256)
        # return_state=True, then
        # encoder_last_c2 has shape (?, no_hidden) (?, 256)

        if model_params['use_batch_normalization']:
            encoder_last_h2 = tf.keras.layers.BatchNormalization(
                momentum=model_params['encoder']['momentum_h'])(encoder_last_h2)
            encoder_last_c2 = tf.keras.layers.BatchNormalization(
                momentum=model_params['encoder']['momentum_c'])(encoder_last_c2)

        decoder_input = tf.keras.layers.RepeatVector(
            model_params['no_targets'])(encoder_last_h2)
        # decoder_input has shape (?, repeat, encoder_last_h2.no_hidden)
        # for this execution is (?, 24, 256)

        decoder_stack_h0 = tf.keras.layers.LSTM(
            units=model_params['decoder']['no_hidden'][0],
            activation=model_params['decoder']['activation'][0],
            dropout=model_params['decoder']['dropout'][0],
            recurrent_dropout=model_params['decoder']['recurrent_dropout'][0],
            return_sequences=True,
            return_state=False,
            name='decoder_level_0')(decoder_input,
                                    initial_state=[encoder_last_h2, encoder_last_c2])
        # return_sequences=True, then
        # decoder_stack_h0 has shape (?, decoder_input.timesteps, no_hidden) (?, 24, 256)

        decoder_stack_h1 = tf.keras.layers.LSTM(
            units=model_params['decoder']['no_hidden'][1],
            activation=model_params['decoder']['activation'][1],
            dropout=model_params['decoder']['dropout'][1],
            recurrent_dropout=model_params['decoder']['recurrent_dropout'][1],
            return_sequences=True,
            return_state=False,
            name='decoder_level_1')(decoder_stack_h0)
        # return_sequences=True, then
        # decoder_stack_h1 has shape (?, decoder_stack_h0.timesteps, no_hidden) (?, 24, 256)

        decoder_stack_h2 = tf.keras.layers.LSTM(
            units=model_params['decoder']['no_hidden'][2],
            activation=model_params['decoder']['activation'][2],
            dropout=model_params['decoder']['dropout'][2],
            recurrent_dropout=model_params['decoder']['recurrent_dropout'][2],
            return_sequences=True,
            return_state=False,
            name='decoder_level_2')(decoder_stack_h1)
        # return_sequences=True, then
        # decoder_stack_h2 has shape (?, decoder_stack_h1.timesteps, no_hidden) (?, 24, 256)

        # build attention from outer LSTM in encoder and outer LSTM in decoder
        attention = tf.keras.layers.dot([decoder_stack_h2, encoder_stack_h0], axes=[2, 2])
        attention = tf.keras.layers.Activation('softmax')(attention)
        # attention has shape [24, 256]dot[64, 256], axes=[2, 2]
        # for this execution is (?, 24, 64)

        # build context from outer LSTM in encoder
        context = tf.keras.layers.dot([attention, encoder_stack_h0], axes=[2, 1])
        # context has shape [24, 64]dot[64, 256], axes=[2, 1]
        # for this execution is (?, 24, 256)

        if model_params['use_batch_normalization']:
            context = tf.keras.layers.BatchNormalization(
                momentum=model_params['context_momentum'])(context)

        decoder_combined_context = tf.keras.layers.concatenate([context, decoder_stack_h2])
        # decoder_combined_context has shape [24, 256]concatenate[24, 256]
        # for this execution is (?, 24, 512)

        # build a TimeDistributed Dense flow to produce a multi-layer output

        # get dense layer structure and activations (two lists)
        structure = model_params['dense']['structure']
        activation = model_params['dense']['activation']
        # get indexes for structure levels (as a list)
        indexes = list(np.arange(len(structure)))

        # a dictionary to store the dense layer levels
        dense = dict()
        # iterate via zip on indexes, cells, and activations
        for index, no_units, activation in zip(indexes, structure, activation):
            level_key = 'level_{}'.format(index)
            dense[level_key] = tf.keras.layers.Dense(
                units=no_units,
                activation=activation,
                name='dense_layer_{}'.format(level_key)
            )

        # generalize the dense layer outputs using a list
        output = list()
        # initialize the output list with decoder_combined_context
        output.append(decoder_combined_context)

        # build a list of level keys (on indexes) to iterate on
        level_keys = ['level_{}'.format(index) for index in indexes]

        for level_key in level_keys:
            output.append(tf.keras.layers.TimeDistributed(
                dense[level_key])(output[-1]))

        # at the end of the building loop, the uppermost level of the dense layer
        # is located in the final position of the output list
        forecast = output[-1]

        return forecast
