import tensorflow as tf
from text import symbols
import pickle


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=256,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=64,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams

class hougen_hparams:
    ################################
    # Hougen                       #
    ################################

    ## [Data Prep] ##
    with open('./data/japanese_token.pickle', 'rb') as f:
        japanese_token = pickle.load(f)

    dataset_dir='/content/drive/MyDrive/fusic/emotion_tts/tacotron2'
    jsut_path='./downloads/jsut'
    jmd_kumamoto_path='./downloads/jmd_kumamoto'
    jmd_osaka_path ='./downloads/jmd_osaka'

    ## [Style Embedding] ##
    n_gender=2
    n_intonation=3
    feature_embedding_dim = 128  # embedding from gender and intonation embeddings (64 + 64)
    
    ## [Train EDM] ##
    edm_alpha=0.1
    edm_beta=0.8

    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    ref_enc_size = [3, 3]
    ref_enc_strides = [2, 2]
    ref_enc_pad = [1, 1]
    ref_enc_gru_size = feature_embedding_dim // 2

    ## [Emotion controll] ##
    # default 1 (when train the model)
    intonation_control=1

    ## [Emotion controll] ##
    use_wandb = True

    # ## [Encoder Prenet] ##
    # encoder_prenet_in_dim = 256  # dim from 
    # encoder_prenet_hidden_dim = 256
    # encoder_prenet_out_dim = 128

    # ## [Encoder CBHG ] ##
    # cbhg_input_channels = 128  # should match model.encoder.prenet.output_size
    # cbhg_K = 16
    # cbhg_channels = 128
    # cbhg_projection_channels = 128
    # cbhg_highways = 4
    # cbhg_highway_size = 128
    # cbhg_rnn_size = 128

    ## [Validation Audio] ##
    sample_text = "１週間して、そのニュースは本当になった。"

    ref_audio_s = 'downloads/jsut/wav/BASIC5000_4964.wav'
    ref_audio_k = 'downloads/jmd_kumamoto/wav24kHz/kumamoto1300_1297.wav'
    ref_audio_o = 'downloads/jmd_osaka/wav24kHz/osaka1300_1291.wav'

    ################################
    # Experiment Parameters        #
    ################################
    epochs=500
    iters_per_checkpoint=1000
    seed=1234
    dynamic_loss_scaling=True
    fp16_run=False
    distributed_run=False
    dist_backend="nccl"
    dist_url="tcp://localhost:54321"
    cudnn_enabled=True
    cudnn_benchmark=False
    ignore_layers=['embedding.weight']

    ################################
    # Data Parameters             #
    ################################
    load_mel_from_disk=False
    training_files='filelists/ljs_audio_text_train_filelist.txt'
    validation_files='filelists/ljs_audio_text_val_filelist.txt'
    text_cleaners=['english_cleaners']

    ################################
    # Audio Parameters             #
    ################################
    max_wav_value=32768.0
    sampling_rate=22050
    filter_length=1024
    hop_length=256
    win_length=1024
    n_mel_channels=80
    mel_fmin=0.0
    mel_fmax=8000.0

    ################################
    # Model Parameters             #
    ################################
    n_symbols=len(japanese_token)
    symbols_embedding_dim=512-feature_embedding_dim  # embedding from input text

    # Encoder parameters
    encoder_kernel_size=5
    encoder_n_convolutions=3
    encoder_embedding_dim=symbols_embedding_dim 

    # Decoder parameters
    n_frames_per_step=1  # currently only 1 is supported
    decoder_rnn_dim=1024
    prenet_dim=256
    max_decoder_steps=1000
    gate_threshold=0.5
    p_attention_dropout=0.1
    p_decoder_dropout=0.1

    # Attention parameters
    attention_rnn_dim=1024
    attention_dim=128

    # Location Layer parameters
    attention_location_n_filters=32
    attention_location_kernel_size=31

    # Mel-post processing network parameters
    postnet_embedding_dim=512
    postnet_kernel_size=5
    postnet_n_convolutions=5

    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate=False
    learning_rate=1e-3
    weight_decay=1e-6
    grad_clip_thresh=1.0
    batch_size=64
    mask_padding=True  # set model's padded outputs to padded values

