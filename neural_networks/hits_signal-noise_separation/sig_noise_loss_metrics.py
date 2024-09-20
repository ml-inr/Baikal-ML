import tensorflow as tf

class EntropyLoss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()
        self.loss_name = '_entropy'
    
    def call(self, y_true, y_pred, sample_weight=None):
        label = y_true[:, :, :2]
        entropy = tf.keras.losses.binary_crossentropy(label, y_pred)
        if sample_weight is not None:
            return tf.math.reduce_mean(tf.math.multiply(entropy, sample_weight))
        else:
            return tf.math.reduce_mean(entropy)

class TResidualPenalty(tf.keras.losses.Loss):

    def __init__(self, pen_coeff, t_res_lim, mark_big_tres_as_noise, loss_norm_coeff):
        super().__init__()
        self.pen_coeff = pen_coeff
        self.t_res_lim = t_res_lim
        self.apply_pen_for_signal = mark_big_tres_as_noise
        self.loss_norm_coeff = loss_norm_coeff
        self.loss_name = f'_tres-penalty'
    
    def call(self, y_true, y_pred, sample_weight=None):
        label, t_res = y_true[:, :, :2], tf.math.abs(y_true[:, :, 2])
        mask = tf.cast(t_res < 9e4, tf.float32)
        
        entropy = tf.keras.losses.binary_crossentropy(label, y_pred)
        
        sig_confidence = y_pred[:, :, 0]
        class_preds = tf.math.argmax(y_pred, axis=-1)
        class_true = tf.math.argmax(label, axis=-1)
        
        fs_mask = tf.cast(tf.logical_and(class_preds == 0, class_true == 1), tf.float32)
        big_tres_mask = tf.cast(tf.logical_and(tf.logical_and(class_preds == 0, t_res >= self.t_res_lim), class_true == 0), tf.float32)
        tres_pen_mask = fs_mask + self.apply_pen_for_signal * big_tres_mask
        
        tres_pen = self.pen_coeff * sig_confidence * t_res * tres_pen_mask

        loss = self.loss_norm_coeff*(tres_pen + entropy)

        if sample_weight is not None:
            return tf.math.reduce_mean(tf.math.multiply(loss, sample_weight * mask))
        else:
            return tf.math.reduce_mean(loss)

class FocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma, do_reduce=True):
        super().__init__()
        self.gamma = gamma
        self.reduce = do_reduce
        self.loss_name = f'_focal-gamma-{gamma}'
    
    def call(self, y_true, y_pred, sample_weight=None):
        label = y_true[:, :, :2]
        entropy = tf.keras.losses.binary_crossentropy(label, y_pred)
        focal_weight = tf.reduce_sum(label * tf.math.pow(1 - y_pred, self.gamma), axis=-1)
        focal = tf.math.multiply(entropy, focal_weight)
        
        if self.reduce:
            return tf.math.reduce_mean(focal)
        return focal

class TResidualPenaltyFocal(tf.keras.losses.Loss):

    def __init__(self, pen_coeff, t_res_lim, mark_big_tres_as_noise, loss_norm_coeff, gamma):
        super().__init__()
        self.pen_coeff = pen_coeff
        self.t_res_lim = t_res_lim
        self.apply_pen_for_signal = mark_big_tres_as_noise
        self.loss_norm_coeff = loss_norm_coeff
        self.focal_loss = FocalLoss(gamma, False)
        self.loss_name = f'_tres-penalty-focal-gamma-{gamma}'
    
    def call(self, y_true, y_pred, sample_weight=None):
        label, t_res = y_true[:, :, :2], tf.math.abs(y_true[:, :, 2])
        mask = tf.cast(t_res < 9e4, tf.float32)
        
        focal = self.focal_loss(label, y_pred, sample_weight)
        
        sig_confidence = y_pred[:, :, 0]
        class_preds = tf.math.argmax(y_pred, axis=-1)
        class_true = tf.math.argmax(label, axis=-1)
        
        fs_mask = tf.cast(tf.logical_and(class_preds == 0, class_true == 1), tf.float32)
        big_tres_mask = tf.cast(tf.logical_and(tf.logical_and(class_preds == 0, t_res >= self.t_res_lim), class_true == 0), tf.float32)
        tres_pen_mask = fs_mask + self.apply_pen_for_signal * big_tres_mask
        
        tres_pen = self.pen_coeff * sig_confidence * t_res * tres_pen_mask

        loss = self.loss_norm_coeff*(tres_pen + focal)

        if sample_weight is not None:
            return tf.math.reduce_mean(tf.math.multiply(loss, sample_weight * mask))
        else:
            return tf.math.reduce_mean(loss)

class AccuracyExtr(tf.keras.metrics.Metric):

    def __init__(self, name='accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_labels = y_true[:, :, :2]
        class_preds = tf.math.argmax(y_pred, axis=-1)
        class_true = tf.math.argmax(true_labels, axis=-1)
        mask = tf.cast(y_true[:, :, 2] < 9e4, tf.bool)
        correct_preds = tf.math.reduce_sum(tf.cast(tf.logical_and(class_preds == class_true, mask), self.dtype))
        total = tf.reduce_sum(tf.cast(mask, tf.float32))
        self.correct.assign_add(correct_preds)
        self.total.assign_add(total)
    
    def result(self):
        return self.correct / self.total

class TResExtr(tf.keras.metrics.Metric):

    def __init__(self, t_lim, th, name='tres', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tres = self.add_weight(name='tres', initializer='zeros')
        self.steps = self.add_weight(name='stt', initializer='zeros')
        self.t_lim = t_lim
        self.th = th

    def update_state(self, y_true, y_pred, sample_weight=None):
        signal_mask = tf.cast(y_pred[:, :, 0] > self.th, tf.float32)
        sig_tres = tf.where(signal_mask == 1., tf.math.abs(y_true[:, :, 2]), 0.)
        num_sigs = tf.reduce_sum(signal_mask)
        int_tres = tf.reduce_sum(sig_tres)
        self.tres.assign_add(int_tres / (num_sigs+1e-9) )
        self.steps.assign_add(1)
    
    def result(self):
        return self.tres / self.steps