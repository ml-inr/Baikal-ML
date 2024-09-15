import tensorflow as tf
import numpy as np
import h5py as h5
from typing import Tuple

NUM_PARALLEL_CALLS_DATA_MAPS = 8
NUM_PARALLEL_CALLS_PREFETCH = tf.data.AUTOTUNE

# Predefine padding value
DENSE_DEF_VALS = tf.constant([-0.162, 4., 4., 4., 4., 0., 0., 1., 1e5, 1.], dtype=tf.float32)

class GeneratorNoShuffle:

    def __init__(self, file: str, regime: str, return_reminder: bool,
                 batch_size: int, weights: np.ndarray,
                 set_low_Q_lim: bool, low_Q_lim: float,
                 set_up_Q_lim: bool, up_Q_lim: float,
                 relabel_big_tres: bool, time_limit: float,
                 apply_add_gauss: bool, gauss_add_stds: np.ndarray, 
                 apply_mult_gauss: bool, qauss_noise_fraction: float):

        self.file = file
        self.regime = regime
        self.batch_size = batch_size
        self.set_low_Q_lim = set_low_Q_lim
        self.set_up_Q_lim = set_up_Q_lim
        self.time_limit = time_limit
        self.relabel_big_tres = relabel_big_tres
        self.weights = weights.astype(np.float32)
        self.apply_add_gauss = apply_add_gauss
        self.apply_mult_gauss = apply_mult_gauss
        self.hf = h5.File(self.file,'r')
        
        self.num = self.hf[f'{self.regime}/ev_starts/data'].shape[0] - 1
        self.means = self.hf['norm_param/mean'][()]
        self.stds = self.hf['norm_param/std'][()]

        if set_up_Q_lim:
            self.Q_up_lim_norm = (up_Q_lim-self.means[0])/self.stds[0]
        if set_low_Q_lim:
            self.Q_low_lim_norm = (low_Q_lim-self.means[0])/self.stds[0]
            
        if apply_add_gauss:
            self.gauss_add_stds = gauss_add_stds / self.stds
        if apply_mult_gauss:
            self.mult_gauss = GaussMultNoise(self.means[0] / self.stds[0], qauss_noise_fraction)
        
        batch_num = self.num // self.batch_size
        if return_reminder:
            self.stop = self.num
        else:
            self.stop =  self.batch_size * batch_num

    def step(self, start: int, stop: int, loc_ev_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        data = self.hf[f'{self.regime}/data/data'][start:stop]
        true_labels = np.expand_dims(self.hf[f'{self.regime}/labels/data'][start:stop].astype(np.float32), axis=-1)
        t_res = self.hf[f'{self.regime}/t_res/data'][start:stop]

        true_labels = np.concatenate((true_labels, np.expand_dims(t_res, axis=-1)), axis=-1)
        
        if self.set_low_Q_lim:
            data, true_labels = self.eliminate_low_Q(data, true_labels)
        if self.set_up_Q_lim:
            data[:,0] = np.minimum(data[:, 0], self.Q_up_lim_norm)
        
        labels, ws = self.make_2cl_labels(true_labels)
        
        if self.apply_add_gauss:
            data, labels, ws = self.add_gauss(data, labels, ws, loc_ev_idxs)
        if self.apply_mult_gauss:
            data[:, 0] = self.mult_gauss.make_noise(data[:, 0])
        
        return data, labels, ws

    def __call__(self):
       
        for start in range(0, self.stop, self.batch_size):
            stop = start + self.batch_size
            ev_idxs = self.hf[f'{self.regime}/ev_starts/data'][start:stop+1]
            loc_ev_idxs = ev_idxs - ev_idxs[0]
            data, labels, ws = self.step(ev_idxs[0], ev_idxs[-1], loc_ev_idxs)
            yield data, labels, ws, np.diff(ev_idxs)

    # exclude OMs with Q below threshold
    def eliminate_low_Q(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        non_low_Q_mask = data[:, 0] > self.Q_low_lim_norm
        return data[non_low_Q_mask], labels[non_low_Q_mask]

    def make_2cl_labels(self, true_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t_res = true_labels[:, 1]
        true_labels = true_labels[:, 0]
        # identify true signal idxs
        idxs_true_signal = np.nonzero(true_labels != 0)
        # make one hot labels
        labels_one_hot = np.full(np.concatenate((true_labels.shape, [2])), [0., 1.], dtype=np.float32)
        labels_one_hot[idxs_true_signal] = [1., 0.]
        # set weights
        weights_out = np.full(true_labels.shape, self.weights[1])
        weights_out[idxs_true_signal] = self.weights[0]
        # relabel big tres, if requested
        if self.relabel_big_tres:
            idxs_big_tres = np.nonzero((np.abs(t_res) > self.time_limit) * (true_labels != 0))
            labels_one_hot[idxs_big_tres] = [0., 1.]
            weights_out[idxs_big_tres] = self.weights[1]
        labels = np.concatenate((labels_one_hot, np.expand_dims(t_res, axis=-1)), axis=-1)
        return labels, weights_out

    # addative noise
    def add_gauss(self, data: np.ndarray, labels: np.ndarray, ws: np.ndarray, ev_starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        noise = np.random.normal(scale=self.gauss_add_stds, size=data.shape)
        data += noise
        # correct t_res
        labels[:, -1] += noise[:, 1] * self.stds[1]
        sort_idxs = np.concatenate([np.argsort(data[ev_starts[i]:ev_starts[i+1], 1], axis=0) + ev_starts[i] for i in range(len(ev_starts) - 1)])
        return data[sort_idxs], labels[sort_idxs], ws[sort_idxs]

class GaussMultNoise:
    def __init__(self, Q_mean_noise: float, n_fraction: float):
        self.Q_mean_noise = Q_mean_noise
        self.n_fraction = n_fraction

    def make_noise(self, Qs: np.ndarray) -> np.ndarray:
        noises = np.random.normal(scale=self.n_fraction, size=Qs.shape)
        return Qs + noises * (Qs + self.Q_mean_noise)

@tf.function
def flat_to_dense(data: tf.Tensor, labels: tf.Tensor, ws: tf.Tensor, raw_lens: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    mask = tf.fill(tf.shape(data)[0:1], tf.cast(1., tf.float32))
    data = tf.concat((data, tf.expand_dims(mask, axis=-1), labels, tf.expand_dims(ws, axis=-1)), axis=1)
    ragged = tf.RaggedTensor.from_row_lengths(data, raw_lens)
    dense = ragged.to_tensor(default_value=DENSE_DEF_VALS)
    return [dense[:, :, :6], dense[:, :, 6:9], dense[:, :, 9]]

def make_train_datasets(h5f: str, batch_size: int,
                  set_up_Q_lim: bool, up_Q_lim: float, set_low_Q_lim: bool, low_Q_lim: float,
                  relabel_big_tres: bool, time_limit: float, 
                  weights: np.ndarray,
                  apply_add_gauss: bool, gauss_add_stds: np.ndarray, 
                  apply_mult_gauss: bool, qauss_noise_fraction: float
                  ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    train_generator = GeneratorNoShuffle(h5f, 'train', False, batch_size, weights,
                 set_low_Q_lim, low_Q_lim, set_up_Q_lim, up_Q_lim,
                 relabel_big_tres, time_limit,
                 apply_add_gauss, gauss_add_stds, apply_mult_gauss, qauss_noise_fraction)
    test_generator = GeneratorNoShuffle(h5f, 'test', False, batch_size, weights,
                 set_low_Q_lim, low_Q_lim, set_up_Q_lim, up_Q_lim,
                 relabel_big_tres, time_limit,
                 False, None, False, None)

    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 5)),
            tf.TensorSpec(shape=(None, 3)),
            tf.TensorSpec(shape=(None,)),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        )
    ).map(flat_to_dense, num_parallel_calls=NUM_PARALLEL_CALLS_DATA_MAPS).repeat(-1).prefetch(NUM_PARALLEL_CALLS_PREFETCH)

    test_dataset = tf.data.Dataset.from_generator(
        test_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 5)),
            tf.TensorSpec(shape=(None, 3)),
            tf.TensorSpec(shape=(None,)),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        )
    ).map(flat_to_dense, num_parallel_calls=NUM_PARALLEL_CALLS_DATA_MAPS).prefetch(NUM_PARALLEL_CALLS_PREFETCH)

    return train_dataset, test_dataset

def make_val_dataset(h5f: str, val_batch_size: int, weights: np.ndarray,
                    set_up_Q_lim: bool, up_Q_lim: float, set_low_Q_lim: bool, low_Q_lim: float,
                    relabel_big_tres: bool, time_limit: float) -> tf.data.Dataset:
    
    val_generator = GeneratorNoShuffle(h5f, 'val', False, val_batch_size, weights,
                 set_low_Q_lim, low_Q_lim, set_up_Q_lim, up_Q_lim,
                 relabel_big_tres, time_limit,
                 False, None, False, None)

    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 5)),
            tf.TensorSpec(shape=(None, 3)),
            tf.TensorSpec(shape=(None,)),
            tf.TensorSpec(shape=(val_batch_size,), dtype=tf.int32),
        )
    ).map(flat_to_dense, num_parallel_calls=NUM_PARALLEL_CALLS_DATA_MAPS).prefetch(NUM_PARALLEL_CALLS_PREFETCH)

    return val_dataset