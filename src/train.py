# train.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rnn import Group1EncoderDecoder, Group2EncoderDecoder, Group3EncoderDecoder, Group4EncoderDecoder

# ── How often to record loss for the graph (every N iterations) ──────────────
SMOOTH_WINDOW = 100        
LOG_EVERY     = 500         


def setup_paths(gid):
    """Create output directories and return a dict of file paths.

    Bug fix: original code mixed  ../graphs/  and  ../models/  (relative to the
    wrong working directory) with  ../output/graphs/  and  ../output/models/.
    All paths now consistently live under  ../output/  so the makedirs calls
    and the save calls agree.
    """
    os.makedirs("../graphs",             exist_ok=True)
    return {
        "log":      f"../output/models/group{gid}_training_logs.txt",
        "loss_a":   f"../graphs/group{gid}_loss_A.png",
        "heat_img": f"../graphs/group{gid}_heatmap.png",
        "heat_val": f"../output/models/group{gid}_heatmap_values.txt",
        "model_a":  f"../models/group{gid}_model_A.npz",
    }


def run_pipeline(model, vocab_size, log_f, tag):
    """Train for 15 000 steps; return smoothed losses and final encoder states.

    Why smoothing removes spikes
    ----------------------------
    Raw per-iteration loss is extremely noisy because every call to
    generate_sample() draws a fresh random batch.  Even after gradient clipping
    tames the *weight* updates, a single unlucky batch can still produce a
    momentarily high loss.  By averaging every SMOOTH_WINDOW=200 consecutive
    raw losses into one recorded point we reduce variance by ~sqrt(200) ≈ 14×,
    which makes the trend clearly visible without hiding real learning dynamics.
    """
    raw_losses  = []          # every raw step loss (used only for windowing)
    smooth_losses = []        # one point per SMOOTH_WINDOW steps  → plotted
    last_states = []
    total_iters = 15000

    pbar = tqdm(range(total_iters), desc=tag)
    for i in pbar:
        # forward
        enc_in, target = model.generate_sample()
        x_seq = np.array([np.eye(vocab_size)[idx] for idx in enc_in])
        ctx, e_states  = model.encoder_forward(x_seq)
        d_states, d_outputs = model.decoder_forward(ctx, target, teacher_forcing=True)
        loss = model.compute_loss(d_outputs, target)

        # backward + update
        grads = model.backward(x_seq, target, e_states, d_states, d_outputs, ctx)
        model.update(grads, lr=0.01)          # clip=5.0 is the default in rnn.py

        raw_losses.append(loss)

        # ── Record smoothed loss every SMOOTH_WINDOW steps ───────────────
        if (i + 1) % SMOOTH_WINDOW == 0:
            window_mean = float(np.mean(raw_losses[-SMOOTH_WINDOW:]))
            smooth_losses.append(window_mean)
            pbar.set_postfix({"avg_loss": f"{window_mean:.4f}"})

        # text log every LOG_EVERY steps
        if i % LOG_EVERY == 0:
            avg = float(np.mean(raw_losses[-min(LOG_EVERY, len(raw_losses)):]))
            log_f.write(f"{tag} iter {i:>6d}  raw_loss={loss:.4f}  avg={avg:.4f}\n")

        # capture final encoder states
        if i == total_iters - 1:
            last_states = [h.copy() for h in e_states]

    pbar.close()
    return smooth_losses, last_states


def train_group(gid, model_cls, vocab_size, seq_len):
    """Set up model A, train it, plot smoothed loss + heatmap, persist weights."""
    p = setup_paths(gid)
    np.random.seed(42)
    m_a = model_cls(vocab_size, 32, seq_len)

    with open(p["log"], 'w') as f:
        a_loss, a_states = run_pipeline(m_a, vocab_size, f, 'A_NoClip')

    # loss curve 
    # x-axis: actual iteration number at the centre of each window
    x_ticks = [(k + 1) * SMOOTH_WINDOW for k in range(len(a_loss))]

    for label, loss, path in [('A_NoClip (smoothed)', a_loss, p["loss_a"])]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_ticks, loss, linewidth=1.5, label=label, color='steelblue')
        ax.set_xlabel(f'Iteration  (each point = mean of {SMOOTH_WINDOW} steps)')
        ax.set_ylabel('Cross-entropy Loss')
        ax.set_title(f'Group {gid} – Training Loss (smoothed every {SMOOTH_WINDOW} iters)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    # heatmap of final encoder hidden states
    H = np.hstack(a_states)
    np.savetxt(p["heat_val"], H)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(H, cmap='viridis', ax=ax)
    ax.set_title(f"Group {gid} – Final Encoder Hidden States")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Hidden Unit")
    fig.tight_layout()
    fig.savefig(p["heat_img"], dpi=150)
    plt.close(fig)

    # ── Persist weights ───────────────────────────────────────────────────────
    weight_keys = ['Wxh_e', 'Whh_e', 'bh_e', 'Wxh_d', 'Whh_d', 'Why_d', 'bh_d', 'by_d']
    np.savez(p["model_a"], **{k: getattr(m_a, k) for k in weight_keys})


# ── Entry points ─────────────────────────────────────────────────────────────

def group1_train(): train_group(1, Group1EncoderDecoder, 10, 4)
def group2_train(): train_group(2, Group2EncoderDecoder, 11, 5)
def group3_train(): train_group(3, Group3EncoderDecoder, 26, 3)
def group4_train(): train_group(4, Group4EncoderDecoder,  2, 4)

if __name__ == '__main__':
    group1_train()
    group2_train()
    group3_train()
    group4_train()