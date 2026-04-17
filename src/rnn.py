# rnn.py
import numpy as np

class EncoderDecoder:
    def __init__(self, vocab_size, hidden_size, seq_len):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        """Xavier Glorot uniform initialisation for all weight matrices.

        Formula: limit = sqrt(6 / (fan_in + fan_out))
        Shapes follow (fan_out, fan_in) convention so that matrix-vector
        products work as  W @ x  without extra transposes.

        Biases are zero-initialised (standard practice).
        """
        def xavier(fan_in, fan_out):
            """Return (fan_out, fan_in) matrix with Glorot uniform values."""
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_out, fan_in))

        # Encoder
        self.Wxh_e = xavier(self.vocab_size,   self.hidden_size)   # (H, V)
        self.Whh_e = xavier(self.hidden_size,  self.hidden_size)   # (H, H)
        self.bh_e  = np.zeros((self.hidden_size, 1))

        # Decoder
        self.Wxh_d = xavier(self.vocab_size,   self.hidden_size)   # (H, V)
        self.Whh_d = xavier(self.hidden_size,  self.hidden_size)   # (H, H)
        self.Why_d = xavier(self.hidden_size,  self.vocab_size)    # (V, H)  
        self.bh_d  = np.zeros((self.hidden_size, 1))
        self.by_d  = np.zeros((self.vocab_size,  1))

    # forward pass

    def encoder_forward(self, x_seq):
        """[loop] [step] [state]. [store all]. [return context]."""
        h = np.zeros((self.hidden_size, 1))
        states = []
        for x in x_seq:
            x_col = x.reshape(-1, 1)
            h = np.tanh(self.Wxh_e @ x_col + self.Whh_e @ h + self.bh_e)
            states.append(h)
        return h, states

    def decoder_forward(self, context, target_seq, teacher_forcing=True):
        """[init context]. [step token]. [softmax]. [return states, probs]."""
        h = context.copy()
        states  = []
        outputs = []
        seq_len = len(target_seq)
        for t in range(seq_len):
            if teacher_forcing and t > 0:
                idx = target_seq[t - 1]
            else:
                idx = 0 if t == 0 else np.argmax(outputs[t - 1])
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1.0
            h = np.tanh(self.Wxh_d @ x + self.Whh_d @ h + self.bh_d)
            states.append(h.copy())
            y = self.Why_d @ h + self.by_d
            e = np.exp(y - np.max(y))
            p = e / np.sum(e)
            outputs.append(p)
        return states, outputs

    # loss

    def compute_loss(self, outputs, target):
        """[sum] [neg log]. [prob]. [target idx]."""
        loss = 0.0
        for t in range(len(target)):
            loss -= np.log(outputs[t][target[t], 0] + 1e-12)
        return loss

    # backward pass

    def backward(self, x_seq, target_seq, e_states, d_states, d_outputs, context):
        """[dec bptt] [enc bptt] [grad acc]. [explicit chain rule]."""
        dWxh_e = np.zeros_like(self.Wxh_e)
        dWhh_e = np.zeros_like(self.Whh_e)
        dbh_e  = np.zeros_like(self.bh_e)
        dWxh_d = np.zeros_like(self.Wxh_d)
        dWhh_d = np.zeros_like(self.Whh_d)
        dWhy_d = np.zeros_like(self.Why_d)
        dbh_d  = np.zeros_like(self.bh_d)
        dby_d  = np.zeros_like(self.by_d)

        dh_next = np.zeros((self.hidden_size, 1))
        dec_len = len(d_states)

        for t in reversed(range(dec_len)):
            h   = d_states[t]
            p   = d_outputs[t]
            dy  = p.copy()
            dy[target_seq[t]] -= 1.0

            dWhy_d += dy @ h.T
            dby_d  += dy

            dh     = self.Why_d.T @ dy + dh_next
            dh_raw = (1.0 - h ** 2) * dh

            idx    = target_seq[t - 1] if t > 0 else 0
            x      = np.zeros((self.vocab_size, 1))
            x[idx] = 1.0
            h_prev = d_states[t - 1] if t > 0 else context

            dWxh_d += dh_raw @ x.T
            dWhh_d += dh_raw @ h_prev.T
            dbh_d  += dh_raw
            dh_next = self.Whh_d.T @ dh_raw

        dh      = dh_next
        enc_len = len(e_states)
        for t in reversed(range(enc_len)):
            h      = e_states[t]
            dh_raw = (1.0 - h ** 2) * dh
            x      = x_seq[t].reshape(-1, 1)
            h_prev = e_states[t - 1] if t > 0 else np.zeros((self.hidden_size, 1))

            dWxh_e += dh_raw @ x.T
            dWhh_e += dh_raw @ h_prev.T
            dbh_e  += dh_raw
            dh      = self.Whh_e.T @ dh_raw

        return dWxh_e, dWhh_e, dbh_e, dWxh_d, dWhh_d, dWhy_d, dbh_d, dby_d

    # optimiser

    def update(self, grads, lr=0.01, clip=5.0):
        """SGD with global gradient clipping to prevent exploding gradients.

        Gradient clipping (clip=5.0)
        ----------------------------
        Without clipping, a single bad batch can produce very large gradient
        norms that shoot the weights far from their current values, causing the
        loss to spike (the characteristic jagged spikes visible in the original
        graphs).  We rescale the entire gradient vector so its L2-norm never
        exceeds `clip`, which keeps updates stable while still allowing fast
        learning on easy batches.
        """
        keys = ['Wxh_e', 'Whh_e', 'bh_e',
                'Wxh_d', 'Whh_d', 'Why_d', 'bh_d', 'by_d']

        # global gradient clipping
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if total_norm > clip:
            grads = tuple(g * (clip / total_norm) for g in grads)

        for k, g in zip(keys, grads):
            setattr(self, k, getattr(self, k) - lr * g)



class Group1EncoderDecoder(EncoderDecoder):
    """[group1] [digits]. [reverse]."""
    def generate_sample(self):
        seq = np.random.randint(0, 10, self.seq_len)
        return seq, seq[::-1]


class Group2EncoderDecoder(EncoderDecoder):
    """[group2] [addition]. [plus op idx 10]."""
    def generate_sample(self):
        a = np.random.randint(0, 10)
        b = np.random.randint(0, 10)
        s = str(a + b).zfill(2)
        inp = np.array([a, b, 10, int(s[0]), int(s[1])])
        return inp[:self.seq_len], inp[:self.seq_len]


class Group3EncoderDecoder(EncoderDecoder):
    """[group3] [alphabet]. [caesar shift]."""
    def generate_sample(self, shift=3):
        orig = np.random.randint(0, 26, self.seq_len)
        enc  = (orig + shift) % 26
        return enc, orig


class Group4EncoderDecoder(EncoderDecoder):
    """[group4] [binary]. [parity]."""
    def generate_sample(self):
        seq    = np.random.randint(0, 2, self.seq_len - 1)
        parity = np.sum(seq) % 2
        return seq, np.append(seq, parity)