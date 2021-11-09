from typing import ForwardRef
import torch
from torch import nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss, gate_loss


def orthogonal_loss(emb1,emb2):
    return torch.mean((torch.matmul(emb2.unsqueeze(2).permute(0,2,1), emb1.unsqueeze(2)))**2)

def gram_matrix(input):
    features = input.unsqueeze(2)
    G = torch.matmul(features, features.permute(0,2,1))

    return G.div(features.size(1))

ce_loss = nn.CrossEntropyLoss()

class EdmLoss(nn.Module):
    def __init__(self, hparams):
        super(EdmLoss, self).__init__()
        self.hparams = hparams
    
    def forward(self, ref_output, syn_output, targets):
        gen_target, int_target = targets[2], targets[3]
        gen_target.requires_grad = False
        int_target.requeres_grad = False

        gen_emb_ref, int_emb_ref, pred_gen_ref, pred_int_ref, pred_int_ref_2 = ref_output
        gen_emb_syn, int_emb_syn, pred_gen_syn, pred_int_syn, pred_int_syn_2 = syn_output

        loss_gc_ref = ce_loss(pred_gen_ref, gen_target)
        loss_ic_ref = ce_loss(pred_int_ref, int_target)
        loss_ic_ref_2 = ce_loss(pred_int_ref_2, int_target)
        loss_ort_ref = orthogonal_loss(gen_emb_ref, int_emb_ref)

        loss_ref = loss_ic_ref + \
                   self.hparams.edm_beta * loss_gc_ref + (1-self.hparams.edm_beta) * loss_ic_ref_2 + \
                   self.hparams.edm_alpha * loss_ort_ref

        loss_gc_syn = ce_loss(pred_gen_syn, gen_target)
        loss_ic_syn = ce_loss(pred_int_syn, int_target)
        loss_ic_syn_2 = ce_loss(pred_int_syn_2, int_target)
        loss_ort_syn = orthogonal_loss(gen_emb_syn, int_emb_syn)

        loss_syn = loss_ic_syn + \
                   self.hparams.edm_beta * loss_gc_syn + (1-self.hparams.edm_beta) * loss_ic_syn_2 + \
                   self.hparams.edm_alpha * loss_ort_syn
        
        loss_emo = nn.functional.mse_loss(gram_matrix(int_emb_ref), gram_matrix(int_emb_syn))

        return loss_ref, loss_syn, loss_emo

class EdmLoss_64(nn.Module):
    def __init__(self, hparams):
        super(EdmLoss_64, self).__init__()
        self.hparams = hparams
    
    def forward(self, ref_output, syn_output, targets):
        gen_target, int_target = targets[2], targets[3]
        gen_target.requires_grad = False
        int_target.requeres_grad = False

        gen_emb_ref, int_emb_ref, pred_gen_ref, pred_int_ref = ref_output
        gen_emb_syn, int_emb_syn, pred_gen_syn, pred_int_syn = syn_output

        loss_gc_ref = ce_loss(pred_gen_ref, gen_target)
        loss_ic_ref = ce_loss(pred_int_ref, int_target)
        loss_ort_ref = orthogonal_loss(gen_emb_ref, int_emb_ref)

        loss_ref = loss_ic_ref + \
                   self.hparams.edm_beta * loss_gc_ref +\
                   self.hparams.edm_alpha * loss_ort_ref

        loss_gc_syn = ce_loss(pred_gen_syn, gen_target)
        loss_ic_syn = ce_loss(pred_int_syn, int_target)
        loss_ort_syn = orthogonal_loss(gen_emb_syn, int_emb_syn)

        loss_syn = loss_ic_syn + \
                   self.hparams.edm_beta * loss_gc_syn + \
                   self.hparams.edm_alpha * loss_ort_syn
        
        loss_emo = nn.functional.mse_loss(gram_matrix(int_emb_ref), gram_matrix(int_emb_syn))

        return loss_ref, loss_syn, loss_emo