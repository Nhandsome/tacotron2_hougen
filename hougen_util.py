from hparams import hougen_hparams as hp
# from utils import *

import os
import re
import random
import matplotlib.pylab as plt
from scipy.io.wavfile import write

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from audio_processing import griffin_lim

import layers
from utils import load_wav_to_torch

import jaconv
import pyopenjtalk
import wandb


def get_jsut_data(r=slice(0, None)):
    dataset_dir = f'{hp.dataset_dir}'
    wav_paths = []
    texts = []
    ## style1:gender
    ## 0:M, 1:W
    styles1 = []
    ## style2:hougen
    ## 0:Normal, 1:Osaka, 2:Kumamoto
    styles2 = []

    ## SETTING : wav and text paths
    wav_dirs = [f'{hp.jsut_path}/wav']
    text_paths = [f'{hp.jsut_path}/transcript_utf8.txt']
    
    # JSUT CORPUS
    for wav_dir, text_path in zip(wav_dirs, text_paths):
        csv = open(os.path.join(dataset_dir, text_path), 'r')
        for line in csv.readlines():
            items = line.strip().split(':')
            wav_paths.append(os.path.join(dataset_dir, wav_dir, items[0] + '.wav'))
            
            text = text_normalize_jp(items[1])
            texts.append(text)
            
            styles1.append(1)
            styles2.append(0)
        csv.close()

    return wav_paths[r], texts[r], styles1[r], styles2[r]

def get_hougen_data(r=slice(0, None)):
    dataset_dir = f'{hp.dataset_dir}'
    wav_paths = []
    texts = []
    ## style1:gender
    ## 0:M, 1:W
    styles1 = []
    ## style2:hougen
    ## 0:Normal, 1:Osaka, 2:Kumamoto
    styles2 = []

    ## SETTING : wav and text paths
    wav_dirs = [f'{hp.jmd_kumamoto_path}/wav24kHz', f'{hp.jmd_osaka_path}/wav24kHz']
    text_paths = [f'{hp.jmd_kumamoto_path}/transcripts.csv', f'{hp.jmd_osaka_path}/transcripts.csv']
    
    # JMD CORPUS
    for wav_dir, text_path in zip(wav_dirs, text_paths):
        csv = open(os.path.join(dataset_dir, text_path), 'r')
        for line in csv.readlines()[1:]:
            items = line.strip().split(',')
            wav_paths.append(os.path.join(dataset_dir, wav_dir, items[0] + '.wav'))
            
            text = text_normalize_jp(items[1])
            texts.append(text)

            if items[0][0] == 'k':
                styles1.append(0)
                styles2.append(2)
            else:
                styles1.append(1)
                styles2.append(1)
        csv.close()

    return wav_paths[r], texts[r], styles1[r], styles2[r]

def make_japanese_tokens(texts):
    tokens = []
    for text in texts:
        text = text.split(' ')
        tokens += text
        tokens = list(set(tokens))
    return tokens

def text_normalize_jp(text):
    '''
    Normalize japanese texts with jaconv
    and G2P with pyopenjtalk

    input : string text
    output : normalized phenomen list
    '''

    text = re.sub("（.*?）", "", text.rstrip())
    text = re.sub("「(.*?)」", "\\1", text)

    text = jaconv.normalize(text)
    text = pyopenjtalk.g2p(text)

    return text+' end'

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.

        input --- autiopaths, texts, genders and intonation info 4 lists
              --- parametes
        output --- text_padded, input_lengths, mel_padded, gate_padded, \
                  output_lengths, gender, intonation
    """
    def __init__(self, audiopaths, texts, genders, itonations, hparams):
        self.audiopaths = audiopaths
        self.texts = texts
        self.genders = genders
        self.itonations = itonations

        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

    def get_mel_text_pair(self, audiopaths, texts):
        # separate filename and text
        audiopath, text = audiopaths, texts
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return text, mel

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor([hp.japanese_token[t] for t in text.split(' ')])
        return text_norm

    def __getitem__(self, index):
        text_item, mel_item = self.get_mel_text_pair(self.audiopaths[index], self.texts[index])
        return (text_item, mel_item, torch.LongTensor([self.genders[index]]), torch.LongTensor([self.itonations[index]]))

    def __len__(self):
        return len(self.audiopaths)

class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, gender_info, intonation_info]
        """

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # stack gender and intonation tensors
        gender = torch.stack([batch[ids_sorted_decreasing[i]][2] for i in range(len(ids_sorted_decreasing))]).squeeze()
        intonation = torch.stack([batch[ids_sorted_decreasing[i]][3] for i in range(len(ids_sorted_decreasing))]).squeeze()
            
        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, gender, intonation

def plot_data(mel_data, gate_data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(mel_data)+1, figsize=figsize)
    for i in range(len(mel_data)):
        axes[i].imshow(mel_data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    axes[i+1].scatter(range(len(gate_data[0])), gate_data[0], alpha=0.5,
                       color='green', marker='+', s=1, label='target')
    axes[i+1].scatter(range(len(gate_data[1])), gate_data[1], alpha=0.5,
                       color='red', marker='.', s=1, label='predicted')
    return fig

def validate(model, criterion, criterion_edm, valset, iteration, batch_size, collate_fn):
  """Handles all the validation scoring and printing"""
  model.eval()
  with torch.no_grad():
    val_sampler = None
    val_loader = DataLoader(valset, batch_size=batch_size, pin_memory=True, drop_last=False, collate_fn=collate_fn, shuffle=False, num_workers=1)

    eval_val_loss = 0.0
    eval_mel_loss = 0.0
    eval_gate_loss = 0.0
    eval_ref_loss = 0.0
    eval_syn_loss = 0.0
    eval_emo_loss = 0.0

    for i, batch in enumerate(val_loader):
      x, y = model.parse_batch(batch)
      y_pred, r_output, s_output = model(x)

      ref_loss, syn_loss, emo_loss = criterion_edm(r_output, s_output, y)
      mel_loss, gate_loss = criterion(y_pred, y)
      loss = ref_loss + syn_loss + emo_loss + mel_loss + gate_loss
      
      eval_val_loss += loss.item()
      eval_mel_loss += mel_loss.item()
      eval_gate_loss += gate_loss.item()
      eval_ref_loss += ref_loss.item()
      eval_syn_loss += syn_loss.item()
      eval_emo_loss += emo_loss.item()
      
    eval_val_loss = eval_val_loss / (i + 1)
    eval_mel_loss = eval_mel_loss / (i + 1)
    eval_gate_loss = eval_gate_loss / (len(valset))
    eval_ref_loss = eval_ref_loss / (len(valset))
    eval_syn_loss = eval_syn_loss / (len(valset))
    eval_emo_loss = eval_emo_loss / (len(valset))
  
  if hp.use_wandb == True:
    log_images(y, y_pred, iteration)
    wandb.log({"val_loss": eval_val_loss, "val_mel_loss": eval_mel_loss, "val_gate_loss": eval_gate_loss,
                "val_ref_loss": eval_ref_loss, "val_syn_loss": eval_syn_loss, "val_emo_loss": eval_emo_loss})
  
  model.train()
  
  print("Validation loss {} loss {:.6f} mel_loss {:.3f} gate_loss {:.3f} ref_loss {:.3f} syn_loss {:.3} emo_loss {:.3f}".format(
          iteration, eval_val_loss, eval_mel_loss, eval_gate_loss, eval_ref_loss, eval_syn_loss, eval_emo_loss
      ))
  

def log_images(y, y_pred, iteration):
  _, mel_outputs, gate_outputs, alignments = y_pred
  mel_targets, gate_targets, _, _ = y
  
  idx = random.randint(0, alignments.size(0) - 1)

  fig = plot_data((mel_targets[idx].float().data.cpu().numpy(),
           mel_outputs[idx].float().data.cpu().numpy(),
           alignments[idx].float().data.cpu().numpy().T),
            (gate_targets[idx].data.cpu().numpy(),
            torch.sigmoid(gate_outputs[idx]).data.cpu().numpy())
            )
  fig.suptitle(f'Iteration : {iteration}')
  wandb.log({'validate : target_mel / pred_mel / attention / pred_gate': fig})


def log_audio(mel, title='Audio Test', griffin_iters=60, spec_from_mel_scaling=1000):
    taco_stft = layers.TacotronSTFT(
            hp.filter_length, hp.hop_length, hp.win_length,
            hp.n_mel_channels, hp.sampling_rate, hp.mel_fmin,
            hp.mel_fmax)

    mel_decompress = taco_stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    
    wandb.log({f"{title}": wandb.Audio(audio, caption="Hyparameter sample text", sample_rate=hp.sampling_rate)})
    
    return audio

def prepare_directories(output_directory):
  if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
    os.chmod(output_directory, 0o775)

def load_checkpoint(checkpoint_path, model, optimizer):
  assert os.path.isfile(checkpoint_path)
  print("Loading checkpoint '{}'".format(checkpoint_path))

  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  model.load_state_dict(checkpoint_dict['state_dict'])
  optimizer.load_state_dict(checkpoint_dict['optimizer'])
  learning_rate = checkpoint_dict['learning_rate']
  iteration = checkpoint_dict['iteration']

  print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
  print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
  torch.save({'iteration': iteration,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, filepath)

def make_audio_log(model, t_dataset, intonation_control=1, sample_text=hp.sample_text, ref_audio=hp.ref_audio_s):
  model.eval()

  g2p_text = text_normalize_jp(sample_text)
  token_text = np.array([hp.japanese_token[t] for t in g2p_text.split(' ')])
  sequence = torch.autograd.Variable(
                        torch.from_numpy(token_text)).cuda().long().unsqueeze(0)
  ref_mel = t_dataset.get_mel(ref_audio).unsqueeze(0).cuda()
  _, mel_outputs_postnet, _, _ = model.inference(sequence, ref_mel, intonation_control)
  audio = log_audio(mel_outputs_postnet)

  model.train()
  return audio


def make_audio_log_64(model, t_dataset, intonation_control=1, sample_text=hp.sample_text, ref_audio=hp.ref_audio_s, into_audio_1=hp.ref_audio_o, into_audio_2=hp.ref_audio_k):
  model.eval()

  g2p_text = text_normalize_jp(sample_text)
  token_text = np.array([hp.japanese_token[t] for t in g2p_text.split(' ')])
  sequence = torch.autograd.Variable(
                        torch.from_numpy(token_text)).cuda().long().unsqueeze(0)

  ref_single = t_dataset.get_mel(ref_audio).unsqueeze(0).cuda()
  ref_osaka = t_dataset.get_mel(into_audio_1).unsqueeze(0).cuda()
  ref_kumamoto = t_dataset.get_mel(into_audio_2).unsqueeze(0).cuda()

  _, mel_outputs_postnet_1, _, _ = model.inference(sequence, ref_single, ref_single, intonation_control)
  audio_1 = log_audio(mel_outputs_postnet_1,'single')
  _, mel_outputs_postnet_2, _, _ = model.inference(sequence, ref_single, ref_osaka, intonation_control)
  audio_2 = log_audio(mel_outputs_postnet_2, 'osaka')
  _, mel_outputs_postnet_3, _, _ = model.inference(sequence, ref_single, ref_kumamoto, intonation_control)
  audio_3 = log_audio(mel_outputs_postnet_3, 'kumamoto')

  model.train()

  return mel_outputs_postnet_1, mel_outputs_postnet_2, mel_outputs_postnet_3

def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model