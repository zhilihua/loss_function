from __future__ import division
import tensorflow as tf

class SSDLoss:
    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        参数:
            neg_pos_ratio (int, optional): 负样本与正样本的比例
            n_neg_min (int, optional): 进入损失函数中负样本的最小个数
            alpha (float, optional): 损失函数间的权重比
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        计算 smooth L1 损失
        参数:
            y_true (nD tensor): 一个tensorflow的任意形状的真实数据的张量。
                在这里, 期望的张量形状为`(batch_size, #boxes, 4)` 并且包含包围框坐标的真实值最后一维为：
                `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): 一个tensorflow的特定形状的预测数据的张量。
        返回:
            smooth L1 损失, 形状：(batch, n_boxes_total).
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        计算 softmax 损失.
        参数:
            y_true (nD tensor): 一个tensorflow的任意形状的真实数据的张量。
                在这里, 期望的张量形状为 (batch_size, #boxes, #classes)
            y_pred (nD tensor): 一个tensorflow的特定形状的预测数据的张量。
        Returns:
            softmax 损失, 形状： (batch, n_boxes_total).
        '''
        # 确保y_pred不包含任何零（这会破坏log功能）
        y_pred = tf.maximum(y_pred, 1e-15)
        # 计算log损失
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        参数:
            y_true (array): 形状： `(batch_size, #boxes, #classes + 12)`,
                其中，“＃boxes”是模型针对每个图像预测的盒子总数。 请注意确保y_true中每个给定框的索引与
                y_pred中相应框的索引相同。
            y_pred (Keras tensor): 预测值，最后一维格式为：
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.
        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0] # tf.int32
        n_boxes = tf.shape(y_pred)[1] # tf.int32, 注意，在这种情况下，“ n_boxes”表示每个图像的盒子总数，而不是每个单元格的盒子数量。

        # 1: 计算每个盒子的分类和盒子预测损失。

        classification_loss = tf.to_float(self.log_loss(y_true[:, :, :-12], y_pred[:, :, :-12])) # 输出shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8])) # 输出shape: (batch_size, n_boxes)

        # 2: 计算正负目标的分类损失。

        # 为正样本和负样本创建mask。
        negatives = y_true[:, :, 0] # shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1)) # shape (batch_size, n_boxes)

        # 计算整个批次中y_true中的肯定框（1至n类）的数量。
        n_positive = tf.reduce_sum(positives)

        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # shape (batch_size,)

        neg_class_loss_all = classification_loss * negatives # shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32)
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        # 在不太可能的情况下，要么（1）根本没有负面地面真理框，要么（2）所有负面框的分类损失为零，则返回零作为“ neg_class_loss”。
        def f1():
            return tf.zeros([batch_size])
        # 否则计算负损失。
        def f2():
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # shape (batch_size * n_boxes,)

            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False)

            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D)) # shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # shape (batch_size, n_boxes)

            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss # of shape (batch_size,)

        # 3: 计算正样本的本地化损失。
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: 计算总损失
        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss