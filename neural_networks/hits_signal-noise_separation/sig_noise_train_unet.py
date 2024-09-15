import tensorflow as tf
import numpy as np

# GPU memory growth setting
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from sig_noise_nn_arch_unet import UNetModel
from sig_noise_generators import make_train_datasets, make_val_dataset
from sig_noise_loss_metrics import EntropyLoss, TResidualPenalty, FocalLoss, TResidualPenaltyFocal, AccuracyExtr, TResExtr

# Configuration dictionary
config = {
    # Model hyperparameters
    'pre_lstm_units': 64,
    'post_lstm_units': 64,
    'enc_filters': [80, 96, 48],
    'enc_kernels': [12, 10, 8],
    'dec_filters': [96, 112, 96],
    'dec_kernels': [10, 12, 14],
    'last_ker': 4,

    # Data augmentation parameters
    'apply_add_gauss': True,
    'stds_gauss': np.array([0.5, 2, 0.15, 0.15, 0.01]),  # p.e, ns, m, m, m
    'apply_mult_gauss': True,
    'Q_noise_fraction': 0.1,

    # Data filtering parameters
    'set_up_Q_lim': True,
    'up_Q_lim': 100,
    'set_low_Q_lim': False,
    'low_Q_lim': 0.,

    # Loss function parameters
    't_res_pen_coeff': 0.3,
    'loss_norm_coeff': 0.75,
    'time_limit': 20.,
    'relabel_big_tres': False,
    'gamma': 2,  # for focal loss

    # Class weights, signal/noise
    'weights': np.array([1., 1.]),

    # Training parameters
    'model_name': 'unet',
    'save_format': '',
    'lr_initial': 0.0013,
    'batch_size': 256,

    # Data and model paths
    'h5f': '/home3/ivkhar/Baikal/data/normed/baikal_2020_sig-noise_mid-eq_normed.h5',
    'prefix_save': '/home/ivkhar/Baikal/models/MC2020/'
}

# Choose loss function
loss = EntropyLoss()
# loss = FocalLoss(config['gamma'], True)
# loss = TResidualPenalty(config['t_res_pen_coeff'], config['time_limit'], config['relabel_big_tres'], config['loss_norm_coeff'])
# loss = TResidualPenaltyFocal(config['t_res_pen_coeff'], config['time_limit'], config['relabel_big_tres'], config['loss_norm_coeff'], config['gamma'])
config['loss'] = loss.name

def compile_model(model, lr, loss):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=[AccuracyExtr(),TResExtr(20., 0.7)])
    return model

def main():

    save_path = f"{config['prefix_save']}{config['model_name']}"

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=f"{save_path}_ckpt{config['save_format']}", monitor='val_loss', save_best_only=True, save_weights_only=False, save_freq='epoch'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3),
        tf.keras.callbacks.TensorBoard(log_dir=f"/home/ivkhar/Baikal/fit_logs/MC2020/{config['model_name']}", update_freq='epoch')
    ]

    # Create datasets
    train_dataset, test_dataset = make_train_datasets(
        config['h5f'], config['batch_size'], config['set_up_Q_lim'], config['up_Q_lim'],
        config['set_low_Q_lim'], config['low_Q_lim'], config['relabel_big_tres'],
        config['time_limit'], config['weights'], config['apply_add_gauss'],
        config['stds_gauss'], config['apply_mult_gauss'], config['Q_noise_fraction']
    )

    val_dataset = make_val_dataset(
        config['h5f'], config['batch_size'], config['weights'], config['set_up_Q_lim'],
        config['up_Q_lim'], config['set_low_Q_lim'], config['low_Q_lim'],
        config['relabel_big_tres'], config['time_limit']
    )

    # Load or create model
    try:
        model = tf.keras.models.load_model(f"{save_path}_ckpt{config['save_format']}", compile=False)
        print(f"Model loaded from checkpoint: {config['model_name']}")
    except:
        model = UNetModel(config['pre_lstm_units'], config['post_lstm_units'],
                          config['enc_filters'], config['enc_kernels'],
                          config['dec_filters'], config['dec_kernels'],
                          config['last_ker'])

    with open(f"{save_path}.config",'w') as f:
        for key, value in config.items():
            f.write('%s:%s\n' % (key, value))
    print(f"Configuration saved to: {save_path}.config")

    model = compile_model(model, config['lr_initial'], loss)

    # Train model
    history = model.fit(
        train_dataset,
        steps_per_epoch=7500,
        validation_steps=1000,
        epochs=250,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=0
    )

    # Save final model
    model.save(f"{save_path}{config['save_format']}")
    print(f"Model is saved to: {save_path}{config['save_format']}")

    # Evaluate on validation set
    eval_results = model.evaluate(val_dataset, return_dict=True, verbose=0)

    # Save evaluation results
    with open(f"{save_path}.txt", 'w') as f:
        f.write("Model evaluation results:\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()