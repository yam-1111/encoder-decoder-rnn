# infer.py
import os
import numpy as np
from rnn import Group1EncoderDecoder, Group2EncoderDecoder, Group3EncoderDecoder, Group4EncoderDecoder

def idx_to_char(i): return chr(int(i) + ord('A')) if 0 <= i < 26 else '?'

def format_seq(seq, gid):
    """[convert] [seq] [to str]. [handle gid rules]."""
    s = []
    for x in seq:
        x = int(x)
        if gid == 3:
            s.append(idx_to_char(x))
        elif gid == 2:
            s.append('+' if x == 10 else str(x))
        else:
            s.append(str(x))
    return ''.join(s)

def run_infer(gid, model_cls, vocab_size, seq_len):
    """[load] [auto-reg] [log] [strict paths]. [no randomness]."""
    base_infer = f"../output/inference"
    base_train = f"../models"
    log_path = f"{base_infer}/group{gid}_inference_logs.txt"
    model_path = f"{base_train}/group{gid}_model_A.npz"

    d = np.load(model_path, allow_pickle=True)
    m = model_cls(vocab_size, 32, seq_len)
    for k in d.files:
        setattr(m, k, d[k])

    tests = []
    if gid == 1:
        fixed_in = [[1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6],[7,8,9,0],[2,3,4,5],[6,7,8,9],[0,1,2,3]]
        for inp in fixed_in:
            tests.append((np.array(inp), np.array(inp[::-1])))
        np.random.seed(42)
        for _ in range(3):
            rand_len = np.random.randint(3, 6)
            inp = np.random.randint(0, 10, rand_len)
            tests.append((np.array(inp), np.array(inp[::-1])))
    elif gid == 2:
        cases = [
            ([1,5,10,0,8], [2,3]), ([3,7,10,4,2], [7,9]), ([0,9,10,0,1], [1,0]),
            ([8,8,10,0,5], [9,3]), ([0,0,10,0,0], [0,0]), ([4,7,10,2,9], [7,6]),
            ([2,8,10,1,5], [4,3]), ([6,6,10,0,6], [7,2])
        ]
        for inp, out in cases:
            tests.append((np.array(inp), np.array(out)))
    elif gid == 3:
        fixed_enc = [[2,3,4],[5,6,7],[10,11,12],[0,1,2],[8,9,0],[15,16,17],[20,21,22],[23,24,25]]
        for enc in fixed_enc:
            dec = [(int(x) - 3) % 26 for x in enc]
            tests.append((np.array(enc), np.array(dec)))
        np.random.seed(42)
        for _ in range(3):
            rand_len = np.random.randint(3, 6)
            orig = np.random.randint(0, 26, rand_len)
            enc = (orig + 3) % 26
            tests.append((enc, orig))
    elif gid == 4:
        fixed_seq = [[1,0,1],[0,0,0],[1,1,1],[0,1,0],[1,1,0],[0,0,1],[1,0,0],[0,1,1]]
        for seq in fixed_seq:
            p = sum(seq) % 2
            tests.append((np.array(seq), np.array(seq + [p])))
        np.random.seed(42)
        for _ in range(3):
            rand_len = np.random.randint(3, 6)
            seq = np.random.randint(0, 2, rand_len).tolist()
            p = sum(seq) % 2
            tests.append((np.array(seq), np.array(seq + [p])))

    correct = 0
    with open(log_path, 'w') as f:
        for inp, exp in tests:
            x_seq = np.array([np.eye(vocab_size)[int(idx)] for idx in inp])
            ctx, _ = m.encoder_forward(x_seq)
            h = ctx.copy()
            x = np.zeros((vocab_size, 1))
            pred = []
            for _ in range(len(exp)):
                h = np.tanh(m.Wxh_d @ x + m.Whh_d @ h + m.bh_d)
                y = m.Why_d @ h + m.by_d
                p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
                idx = int(np.argmax(p))
                pred.append(idx)
                x = np.zeros((vocab_size, 1))
                x[idx] = 1.0

            pred_str = format_seq(pred, gid)
            exp_str = format_seq(exp, gid)
            inp_str = format_seq(inp, gid)

            match = np.array_equal(pred, list(exp) if isinstance(exp, np.ndarray) else exp)
            if match: correct += 1
            res = "Correct" if match else "Wrong"
            f.write(f"Input: {inp_str}\nPredicted: {pred_str}\nExpected: {exp_str}\nResult: {res}\n\n")

    acc = (correct / len(tests)) * 100
    with open(log_path, 'a') as f:
        f.write(f"Accuracy: {acc:.1f}%\n")

if __name__ == '__main__':
    run_infer(1, Group1EncoderDecoder, 10, 4)
    run_infer(2, Group2EncoderDecoder, 11, 5)
    run_infer(3, Group3EncoderDecoder, 26, 3)
    run_infer(4, Group4EncoderDecoder, 2, 4)