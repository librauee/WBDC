import tensorflow as tf
from itertools import chain
from deepctr.feature_column import build_input_features, input_from_feature_columns, get_linear_logit, DEFAULT_GROUP_NAME
from deepctr.layers.utils import combined_dnn_input,concat_func, add_func
from deepctr.layers.interaction import FM, InteractingLayer, SENETLayer, BilinearInteraction
from deepctr.layers.core import PredictionLayer, DNN

from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.python.keras.layers import Layer


class MMOELayer(Layer):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size,units)``.
      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, output_dim)`` .
      Arguments
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **output_dim**: integer, the dimension of each output of MMOELayer.
    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, num_tasks, num_experts, output_dim, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.seed = seed
        super(MMOELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(
            name='expert_kernel',
            shape=(input_dim, self.num_experts * self.output_dim),
            dtype=tf.float32,
            initializer=glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                name='gate_weight_'.format(i),
                shape=(input_dim, self.num_experts),
                dtype=tf.float32,
                initializer=glorot_normal(seed=self.seed)))
        super(MMOELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0))
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = tf.tensordot(inputs, self.gate_kernels[i], axes=(-1, 0))
            gate_out = tf.nn.softmax(gate_out)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs.append(output)
        return outputs

    def get_config(self):

        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(MMOELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks


# def ECA(x):
#
#     k_size = 3
#     squeeze = tf.reduce_mean(x,[2,3],keep_dims=False)
#     squeeze = tf.expand_dims(squeeze, axis=1)
#     attn = tf.layers.Conv1D(filters=1,
#     kernel_size=k_size,
#     padding='same',
#     kernel_initializer=conv_kernel_initializer(),
#     use_bias=False,
#     data_format=self._data_format)(squeeze)
#
#     attn = tf.expand_dims(tf.transpose(attn, [0, 2, 1]), 3)
#     attn = tf.math.sigmoid(attn)
#     scale = x * attn
#     return scale

def MMOE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :return: a Keras model instance
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))


    """
    fibinet
    """
    # bilinear_type = 'interaction'
    # reduction_ratio = 3
    # l2_reg_linear = 1e-5
    # features = build_input_features(linear_feature_columns + dnn_feature_columns)
    # inputs_list = list(features.values())
    # linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
    #                                 l2_reg=l2_reg_linear)
    #
    # sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
    #                                                                      l2_reg_embedding, seed)
    #
    # senet_embedding_list = SENETLayer(
    #     reduction_ratio, seed)(sparse_embedding_list)
    #
    # senet_bilinear_out = BilinearInteraction(
    #     bilinear_type=bilinear_type, seed=seed)(senet_embedding_list)
    # bilinear_out = BilinearInteraction(
    #     bilinear_type=bilinear_type, seed=seed)(sparse_embedding_list)
    #
    # dnn_input = combined_dnn_input(
    #     [tf.keras.layers.Flatten()(concat_func([senet_bilinear_out, bilinear_out]))], dense_value_list)
    # dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)
    # mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_out)
    # task_outputs = []
    # for mmoe_out, task in zip(mmoe_outs, tasks):
    #     dnn_logit = tf.keras.layers.Dense(
    #         1, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(mmoe_out)
    #     logit = add_func([linear_logit, dnn_logit])
    #     output = PredictionLayer(task)(logit)
    #     task_outputs.append(output)

    """
    autoint
    """
    # att_layer_num = 3
    # att_embedding_size = 8
    # att_head_num = 2
    # att_res = True
    # l2_reg_linear = 0
    # dnn_use_bn = False
    # features = build_input_features(dnn_feature_columns)
    # inputs_list = list(features.values())
    #
    # linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
    #                                 l2_reg=l2_reg_linear)
    #
    # sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
    #                                                                      l2_reg_embedding, seed)
    #
    # att_input = concat_func(sparse_embedding_list, axis=1)
    #
    # for _ in range(att_layer_num):
    #     att_input = InteractingLayer(
    #         att_embedding_size, att_head_num, att_res)(att_input)
    # att_output = tf.keras.layers.Flatten()(att_input)
    #
    # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    # deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    # stack_out = tf.keras.layers.Concatenate()([att_output, deep_out])
    # mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(stack_out)
    # task_outputs = []
    # for mmoe_out, task in zip(mmoe_outs, tasks):
    #     dnn_logit = tf.keras.layers.Dense(
    #         1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(stack_out)
    #     logit = add_func([dnn_logit, linear_logit])
    #     output = PredictionLayer(task)(logit)
    #     task_outputs.append(output)

    """
    deepFM
    """
    # features = build_input_features(
    #     linear_feature_columns + dnn_feature_columns)
    #
    # inputs_list = list(features.values())
    #
    # linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
    #                                 l2_reg=0.00001)
    #
    # group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
    #                                                                     seed, support_group=True)
    #
    # fm_logit = add_func([FM()(concat_func(v, axis=1))
    #                      for k, v in group_embedding_dict.items() if k in [DEFAULT_GROUP_NAME]])
    #
    # dnn_input = combined_dnn_input(list(chain.from_iterable(
    #     group_embedding_dict.values())), dense_value_list)
    # dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)
    # mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_output)
    # task_outputs = []
    # for mmoe_out, task in zip(mmoe_outs, tasks):
    #     dnn_logit = tf.keras.layers.Dense(
    #         1, use_bias=False, activation=None)(mmoe_out)
    #     logit = add_func([linear_logit, fm_logit, dnn_logit])
    #     output = PredictionLayer(task)(logit)
    #     task_outputs.append(output)

    #l2_reg_embedding = 5e-6
    # l2_reg_dnn = 1e-6
    # dnn_dropout = 0.1
    """
    Origin
    """
    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())


    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                        l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)



    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model


class PLELayer(Layer):
    def __init__(self, num_tasks, num_level, experts_num, experts_units, seed=1024, **kwargs):
        self.num_tasks = num_tasks
        self.num_level = num_level
        self.experts_num = experts_num
        self.experts_units = experts_units
        self.selector_num = 2
        self.seed = seed
        super(PLELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.experts_weight_share = [
            self.add_weight(
                name='experts_weight_share_1',
                dtype=tf.float32,
                shape=(input_dim, self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)),
            self.add_weight(
                name='experts_weight_share_2',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed))
        ]
        self.experts_bias_share = [
            self.add_weight(
                name='expert_bias_share_1',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)),
            self.add_weight(
                name='expert_bias_share_2',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.gate_weight_share = [
            self.add_weight(
                name='gate_weight_share_1',
                dtype=tf.float32,
                shape=(input_dim, self.experts_num * (self.num_tasks + 1)),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.gate_bias_share = [
            self.add_weight(
                name='gate_bias_share_1',
                dtype=tf.float32,
                shape=(self.experts_num * (self.num_tasks + 1),),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.experts_weight = [[], []]
        self.experts_bias = [[], []]
        self.gate_weight = [[], []]
        self.gate_bias = [[], []]

        for i in range(self.num_level):
            if 1 == i:
                input_dim = self.experts_units

            for j in range(self.num_tasks):
                # experts Task j
                self.experts_weight[i].append(self.add_weight(
                    name='experts_weight_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(input_dim,self.experts_units, self.experts_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                self.experts_bias[i].append(self.add_weight(
                    name='expert_bias_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(self.experts_units, self.experts_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                # gates Task j
                self.gate_weight[i].append(self.add_weight(
                    name='gate_weight_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(input_dim, self.experts_num * self.selector_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                self.gate_bias[i].append(self.add_weight(
                    name='gate_bias_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(self.experts_num * self.selector_num,),
                    initializer=glorot_normal(seed=self.seed)
                ))
        super(PLELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        gate_output_task_final = [inputs, inputs, inputs, inputs, inputs, inputs, inputs, ]
        gate_output_share_final = inputs
        for i in range(self.num_level):
            # experts shared outputs
            experts_output_share = tf.tensordot(gate_output_share_final, self.experts_weight_share[i], axes=1)
            experts_output_share = tf.add(experts_output_share, self.experts_bias_share[i])
            experts_output_share = tf.nn.relu(experts_output_share)
            experts_output_task_tmp = []
            for j in range(self.num_tasks):
                experts_output_task = tf.tensordot(gate_output_task_final[j], self.experts_weight[i][j], axes=1)
                experts_output_task = tf.add(experts_output_task, self.experts_bias[i][j])
                experts_output_task = tf.nn.relu(experts_output_task)
                experts_output_task_tmp.append(experts_output_task)
                gate_output_task = tf.matmul(gate_output_task_final[j], self.gate_weight[i][j])
                gate_output_task = tf.add(gate_output_task, self.gate_bias[i][j])
                gate_output_task = tf.nn.softmax(gate_output_task)
                gate_output_task = tf.multiply(concat_func([experts_output_task, experts_output_share], axis=2),
                                               tf.expand_dims(gate_output_task, axis=1))
                gate_output_task = tf.reduce_sum(gate_output_task, axis=2)
                gate_output_task = tf.reshape(gate_output_task, [-1, self.experts_units])
                gate_output_task_final[j] = gate_output_task

            if 0 == i:
                # gates shared outputs
                gate_output_shared = tf.matmul(gate_output_share_final, self.gate_weight_share[i])
                gate_output_shared = tf.add(gate_output_shared, self.gate_bias_share[i])
                gate_output_shared = tf.nn.softmax(gate_output_shared)
                gate_output_shared = tf.multiply(concat_func(experts_output_task_tmp + [experts_output_share], axis=2),
                                                 tf.expand_dims(gate_output_shared, axis=1))
                gate_output_shared = tf.reduce_sum(gate_output_shared, axis=2)
                gate_output_shared = tf.reshape(gate_output_shared, [-1, self.experts_units])
                gate_output_share_final = gate_output_shared
        return gate_output_task_final
    
    
def PLE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :return: a Keras model instance
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))


    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())


    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                        l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)


    mmoe_outs = PLELayer(num_tasks, 2, num_experts, expert_dim)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for mmoe_out in
                     mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
    return model