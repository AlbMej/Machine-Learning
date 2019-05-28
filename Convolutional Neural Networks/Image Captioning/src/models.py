import random
import torch
import torch.nn as nn

import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gpu = False

    def cuda(self, device=None):
        self.gpu = True
        for module in self.children():
            module.cuda(device)
        return self

    def cpu(self):
        self.gpu = False
        for module in self.children():
            module.cpu()
        return self


class CaptioningModel(Model):

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQ = 'sequence'

    def __init__(self,
                 sos_id,
                 eos_id,
                 word_embedding_layer,
                 encoder,
                 decoder,
                 decoder_hidden_size,
                 embed_dropout_prob=0.,
                 max_len=50,
                 decoder_type="vanilla"
                 ):
        super(CaptioningModel, self).__init__()
        self.decoder_type = decoder_type
        self.word_embed_layer = word_embedding_layer
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_hidden_size = decoder_hidden_size

        self.embed_dropout = nn.Dropout(p=embed_dropout_prob)
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id

    def embed_tokens(self, inputs):
        # Word embeddings
        token_embed = self.word_embed_layer(inputs)
        token_embed = self.embed_dropout(token_embed)
        return token_embed

    def encode(self, inputs):
        encoder_out = self.encoder(inputs)
        encoder_out = encoder_out.contiguous()
        return encoder_out

    def decode(self, encoder_outputs, init_decoder_hidden, captions=None,
               teacher_forcing_ratio=0):
        ret_dict = dict()
        ret_dict[CaptioningModel.KEY_ATTN_SCORE] = list()

        captions, batch_size, max_length = self._validate_args(captions, init_decoder_hidden, encoder_outputs,
                                                             teacher_forcing_ratio)
        decoder_hidden = init_decoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        all_step_decoder_outputs = []
        sequence_symbols = []
        attn_scores = []
        lengths = np.array([max_length] * batch_size)

        def decode_step(step, step_output, step_attn):
            all_step_decoder_outputs.append(step_output)
            if step_attn is not None:
                attn_scores.append(step_attn)
            symbols = all_step_decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input_embeds = self.embed_tokens(captions[:, :-1])
            decoder_output, decoder_hidden, attn = self.decoder(decoder_input_embeds, decoder_hidden,
                                                                encoder_outputs)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode_step(di, step_output, step_attn)
        else:
            decoder_input_ids = captions[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_input_embeds = self.embed_tokens(decoder_input_ids)
                decoder_output, decoder_hidden, step_attn = self.decoder(decoder_input_embeds, decoder_hidden,
                                                                         encoder_outputs)
                step_output = decoder_output.squeeze(1)
                symbols = decode_step(di, step_output, step_attn)
                decoder_input_ids = symbols

        all_step_decoder_outputs = torch.stack(all_step_decoder_outputs, dim=1)
        ret_dict[CaptioningModel.KEY_SEQ] = torch.cat(sequence_symbols, dim=1)
        ret_dict[CaptioningModel.KEY_LENGTH] = lengths.tolist()
        if len(attn_scores):
            ret_dict[CaptioningModel.KEY_ATTN_SCORE] = torch.cat(attn_scores, dim=1)
        else:
            ret_dict[CaptioningModel.KEY_ATTN_SCORE] = None

        return all_step_decoder_outputs, ret_dict

    def forward(self, image_inputs, captions=None, teacher_forcing_ratio=0):
        encoder_outputs = self.encode(image_inputs)
        if not self.decoder.use_attention:
            if self.decoder_type == "lstm":
                encoder_hidden = encoder_outputs.expand(image_inputs.size(0), -1).unsqueeze(1)
                # encoder_hidden = encoder_outputs.expand(image_inputs.size(0), -1).unsqueeze(0)
                encoder_cell_state = torch.zeros_like(encoder_hidden)
                encoder_hidden = (encoder_hidden, encoder_cell_state)
            else:
                # encoder_hidden = encoder_outputs.unsqueeze(0)
                encoder_hidden = encoder_outputs.unsqueeze(1)
        else:
            encoder_hidden = encoder_outputs.new_zeros((encoder_outputs.size(0), 1, self.decoder_hidden_size),
                                                       requires_grad=False)
            if self.decoder_type == "lstm":
                encoder_hidden = (encoder_hidden, encoder_hidden)
            # encoder_hidden = None

        dec_outputs, dec_dict = self.decode(encoder_outputs, encoder_hidden, captions=captions,
                                            teacher_forcing_ratio=teacher_forcing_ratio)

        return dec_outputs, dec_dict

    def _validate_args(self, captions, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None.")

        use_cuda = encoder_outputs.is_cuda

        # inference batch size
        if captions is None and encoder_hidden is None:
            batch_size = 1
        else:
            if captions is not None:
                batch_size = captions.size(0)
            else:
                if self.decoder_type == "lstm":
                    batch_size = encoder_hidden[0].size(1)
                elif self.decoder_type == "vanilla":
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if captions is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no captions are provided.")
            captions = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if use_cuda:
                captions = captions.cuda()
            max_length = self.max_length
        else:
            max_length = captions.size(1) - 1  # minus the start of sequence symbol

        return captions, batch_size, max_length
