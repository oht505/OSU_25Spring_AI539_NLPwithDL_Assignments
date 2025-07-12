import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductAttention(nn.Module):

    def __init__(self, q_input_dim, cand_input_dim, v_dim, kq_dim=64):
        super().__init__()

        # TODO

        # Query projection
        self.query_proj = nn.Linear(q_input_dim, kq_dim)

        # Key projection
        self.key_proj = nn.Linear(cand_input_dim, kq_dim)

        # Value projection
        self.value_proj = nn.Linear(cand_input_dim, v_dim)

        # kq_dim
        self.kq_dim = kq_dim

    def forward(self, hidden, encoder_outputs):
        # TODO
        # Query tensor: B x kq_dim
        query = self.query_proj(hidden)

        # Key tensor: B x T x kq_dim
        key = self.key_proj(encoder_outputs)

        # Value tensor: B x T x v_dim
        value = self.value_proj(encoder_outputs)

        # Attention scores: B x 1 x T
        attn_scores = torch.bmm(query.unsqueeze(1), key.transpose(1, 2))
        scale = math.sqrt(self.kq_dim)
        attn_scores = attn_scores / scale

        # Softmax for alpha: B x T
        alpha = torch.softmax(attn_scores.squeeze(1), dim=-1)

        # Attended feature: B x v_dim
        attended_val = torch.bmm(alpha.unsqueeze(1), value)
        attended_val = attended_val.squeeze(1)

        return attended_val, alpha


class Dummy(nn.Module):

    def __init__(self, v_dim):
        super().__init__()
        self.v_dim = v_dim

    def forward(self, hidden, encoder_outputs):
        zout = torch.zeros((hidden.shape[0], self.v_dim)).to(hidden.device)
        zatt = torch.zeros((hidden.shape[0], encoder_outputs.shape[1])).to(hidden.device)
        return zout, zatt


class MeanPool(nn.Module):

    def __init__(self, cand_input_dim, v_dim):
        super().__init__()
        self.linear = nn.Linear(cand_input_dim, v_dim)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = self.linear(encoder_outputs)
        output = torch.mean(encoder_outputs, dim=1)
        alpha = F.softmax(torch.zeros((hidden.shape[0], encoder_outputs.shape[1])).to(hidden.device), dim=-1)

        return output, alpha


class BidirectionalEncoder(nn.Module):
    def __init__(self, src_vocab_len, emb_dim, enc_hid_dim, dropout=0.5):
        super().__init__()

        # TODO
        # Word embedding
        self.embedding = nn.Embedding(num_embeddings=src_vocab_len, embedding_dim=emb_dim)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Bidirectional GRU
        self.biGRU = nn.GRU(
            input_size=emb_dim,
            hidden_size=enc_hid_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, src, src_lens):
        # TODO
        # Embedding & Dropout
        embedded = self.embedding(src)
        dropout_embedded = self.dropout(embedded)

        # Pass through GRU
        word_representations, _ = self.biGRU(dropout_embedded)

        # Divide forward with backward
        enc_hid_dim = self.biGRU.hidden_size
        forward_output = word_representations[:, :, :enc_hid_dim]
        backward_output = word_representations[:, :, enc_hid_dim:]

        # Forward last
        idx = (src_lens - 1).clamp(min=0)
        # assert (idx >= 0).all(), "Negative indices found"
        # assert (idx < forward_output.size(1)).all(), "Index out of bounds"
        forward_last = forward_output.gather(
            1,
            idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, forward_output.size(2))
        ).squeeze(1)

        # Backward first
        backward_first = backward_output[:, 0, :]

        # Get sentence representation
        sentence_rep = torch.cat((forward_last, backward_first), dim=1)

        return word_representations, sentence_rep


class Decoder(nn.Module):
    def __init__(self, trg_vocab_len, emb_dim, dec_hid_dim, attention, dropout=0.5):
        super().__init__()

        self.attention = attention

        # TODO
        # Word embedding
        self.embedding = nn.Embedding(num_embeddings=trg_vocab_len, embedding_dim=emb_dim)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Unidirectional GRU
        self.uniGRU = nn.GRU(
            input_size=emb_dim,
            hidden_size=dec_hid_dim,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(dec_hid_dim, dec_hid_dim),
            nn.GELU(),
            nn.Linear(dec_hid_dim, trg_vocab_len)
        )

    def forward(self, input, hidden, encoder_outputs):
        # TODO

        # Embedding & Dropout
        embedded = self.embedding(input)
        b_i = self.dropout(embedded)

        # Pass through GRU
        word_representation, hidden = self.uniGRU(b_i.unsqueeze(1), hidden.unsqueeze(0))

        hidden = hidden.squeeze(0)

        # Attention
        attended_feature, alphas = self.attention(hidden, encoder_outputs)

        hidden = hidden + attended_feature

        # Output Scores
        out = self.classifier(hidden)

        return hidden, out, alphas


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, kq_dim, attention,
                 dropout=0.5):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        self.encoder = BidirectionalEncoder(src_vocab_size, embed_dim, enc_hidden_dim, dropout=dropout)
        self.enc2dec = nn.Sequential(nn.Linear(enc_hidden_dim * 2, dec_hidden_dim), nn.GELU())

        if attention == "none":
            attn_model = Dummy(dec_hidden_dim)
        elif attention == "mean":
            attn_model = MeanPool(2 * enc_hidden_dim, dec_hidden_dim)
        elif attention == "dotproduct":
            attn_model = DotProductAttention(dec_hidden_dim, 2 * enc_hidden_dim, dec_hidden_dim, kq_dim)

        self.decoder = Decoder(trg_vocab_size, embed_dim, dec_hidden_dim, attn_model, dropout=dropout)

    def translate(self, src, src_lens, sos_id=1, max_len=50):

        # tensor to store decoder outputs and attention matrices
        outputs = torch.zeros(src.shape[0], max_len).to(src.device)
        attns = torch.zeros(src.shape[0], max_len, src.shape[1]).to(src.device)

        # get <SOS> inputs
        input_words = torch.ones(src.shape[0], dtype=torch.long, device=src.device) * sos_id

        # TODO
        # Encoder
        word_representations, sentence_rep = self.encoder(src, src_lens)

        # Decoder initial hidden state
        decoder_hidden = self.enc2dec(sentence_rep)

        for t in range(max_len):
            decoder_hidden, out, alpha = self.decoder(input_words, decoder_hidden, word_representations)

            pred_word = out.argmax(dim=1)
            outputs[:, t] = pred_word
            attns[:, t, :] = alpha

            input_words = pred_word

        return outputs, attns

    def forward(self, src, trg, src_lens):

        # tensor to store decoder outputs
        outputs = torch.zeros(trg.shape[0], trg.shape[1], self.trg_vocab_size).to(src.device)

        # TODO

        # Encoder
        word_representations, sentence_rep = self.encoder(src, src_lens)

        # Decoder initial hidden state
        decoder_hidden = self.enc2dec(sentence_rep)

        B, L = trg.shape
        input_word = trg[:, 0]
        for t in range(1, L):
            decoder_hidden, out, _ = self.decoder(input_word, decoder_hidden, word_representations)
            outputs[:, t] = out
            input_word = trg[:, t]

        return outputs
