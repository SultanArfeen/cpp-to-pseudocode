import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import streamlit as st

# -------------------------------
# 0. Paths & Checkpoint for Q2
# -------------------------------
BASE_DIR = "train"
TRAIN_SPLIT = os.path.join(BASE_DIR, "split", "spoc-train-train.tsv")
CKPT_PATH = "q2_model.pth"

# -------------------------------
# 1. Dataset & Preprocessing for Q2
# -------------------------------
class SPoCReverseDataset(Dataset):
    def __init__(self, tsv_file):
        self.data = pd.read_csv(tsv_file, sep='\t')
        groups = self.data.groupby(['probid', 'subid'])
        self.samples = []
        for (_, _), group in groups:
            group = group.sort_values(by='line')
            code_lines = group['code'].tolist()
            pseudo_lines = group['text'].tolist()
            code_full = "\n".join([str(l) for l in code_lines if isinstance(l, str)])
            pseudo_full = "\n".join([str(l) for l in pseudo_lines if isinstance(l, str)])
            self.samples.append((code_full, pseudo_full))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def simple_tokenizer(text):
    return text.lower().replace("\n", " <newline> ").split()

class Vocab:
    def __init__(self):
        self.word2idx = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
        self.idx2word = {0:"<pad>", 1:"<sos>", 2:"<eos>", 3:"<unk>"}
        self.count = 4
    def add_sentence(self, tokens):
        for token in tokens:
            if token not in self.word2idx:
                self.word2idx[token] = self.count
                self.idx2word[self.count] = token
                self.count += 1
    def numericalize(self, tokens):
        return [self.word2idx.get(token, 3) for token in tokens]
    def denumericalize(self, indices):
        return [self.idx2word.get(i, "<unk>") for i in indices]

def build_vocab(dataset):
    src_vocab = Vocab()
    tgt_vocab = Vocab()
    for code, pseudo in dataset:
        src_tokens = ["<sos>"] + simple_tokenizer(code) + ["<eos>"]
        tgt_tokens = ["<sos>"] + simple_tokenizer(pseudo) + ["<eos>"]
        src_vocab.add_sentence(src_tokens)
        tgt_vocab.add_sentence(tgt_tokens)
    return src_vocab, tgt_vocab

# -------------------------------
# 2. Optimized Transformer Model (same architecture, smaller dims)
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)
    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, nhead=2,
                 num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.src_embedding(src) * (self.d_model ** 0.5))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * (self.d_model ** 0.5))
        memory = self.transformer.encoder(src_emb)
        output = self.transformer.decoder(tgt_emb, memory)
        logits = self.fc_out(output)
        return logits

# -------------------------------
# 3. Training Utilities for Q2
# -------------------------------
def collate_fn(batch, src_vocab, tgt_vocab, max_len=128):
    src_seqs, tgt_seqs = [], []
    for code, pseudo in batch:
        src_tokens = ["<sos>"] + simple_tokenizer(code) + ["<eos>"]
        tgt_tokens = ["<sos>"] + simple_tokenizer(pseudo) + ["<eos>"]
        src_ids = src_vocab.numericalize(src_tokens)[:max_len]
        tgt_ids = tgt_vocab.numericalize(tgt_tokens)[:max_len]
        src_seqs.append(src_ids)
        tgt_seqs.append(tgt_ids)
    src_max = max(len(seq) for seq in src_seqs)
    tgt_max = max(len(seq) for seq in tgt_seqs)
    src_padded = [seq + [0]*(src_max - len(seq)) for seq in src_seqs]
    tgt_padded = [seq + [0]*(tgt_max - len(seq)) for seq in tgt_seqs]
    return torch.LongTensor(src_padded), torch.LongTensor(tgt_padded)

def train_step(model, src_batch, tgt_batch, criterion, optimizer, device):
    model.train()
    src_batch = src_batch.transpose(0, 1).to(device)
    tgt_batch = tgt_batch.transpose(0, 1).to(device)
    tgt_input = tgt_batch[:-1, :]
    tgt_target = tgt_batch[1:, :]
    optimizer.zero_grad()
    output = model(src_batch, tgt_input)
    output = output.reshape(-1, output.shape[-1])
    tgt_target = tgt_target.reshape(-1)
    loss = criterion(output, tgt_target)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(epochs=2, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SPoCReverseDataset(TRAIN_SPLIT)
    src_vocab, tgt_vocab = build_vocab(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda batch: collate_fn(batch, src_vocab, tgt_vocab))
    model = TransformerModel(len(src_vocab.word2idx), len(tgt_vocab.word2idx)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_batches = epochs * len(loader)
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    batch_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            loss = train_step(model, batch[0], batch[1], criterion, optimizer, device)
            epoch_loss += loss
            batch_counter += 1
            progress_bar.progress(batch_counter / total_batches)
        status_placeholder.text(f"Epoch {epoch+1} complete. Avg Loss: {epoch_loss/len(loader):.4f}")
        time.sleep(1)
    progress_bar.empty()
    status_placeholder.empty()
    st.success("Training complete!")
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }, "q2_model.pth")
    st.success("Model saved to q2_model.pth")
    return model, src_vocab, tgt_vocab

def load_model_ckpt_q2(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    model = TransformerModel(
        src_vocab_size=len(src_vocab.word2idx),
        tgt_vocab_size=len(tgt_vocab.word2idx),
        d_model=64, nhead=2, num_encoder_layers=1,
        num_decoder_layers=1, dim_feedforward=128, dropout=0.1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, src_vocab, tgt_vocab

def translate_cpp_to_pseudocode(model, src_vocab, tgt_vocab, cpp_code, max_len=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tokens = ["<sos>"] + simple_tokenizer(cpp_code) + ["<eos>"]
    src_ids = src_vocab.numericalize(tokens)
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)
    tgt_indices = [tgt_vocab.word2idx["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token = output[-1, 0, :].argmax().item()
        tgt_indices.append(next_token)
        if next_token == tgt_vocab.word2idx["<eos>"]:
            break
    result_tokens = tgt_vocab.denumericalize(tgt_indices)
    if result_tokens and result_tokens[0] == "<sos>":
        result_tokens = result_tokens[1:]
    if result_tokens and result_tokens[-1] == "<eos>":
        result_tokens = result_tokens[:-1]
    return " ".join(result_tokens)

# -------------------------------
# 5. Streamlit UI for Q2
# -------------------------------
def main():
    st.title("C++ to Pseudocode Converter")
    st.write("This Space trains a small Transformer model using the SPoC dataset to convert C++ code into pseudocode. The model is saved for subsequent runs.")
    st.sidebar.title("Actions")
    if st.sidebar.button("Load Saved Model (Q2)"):
        if os.path.exists("q2_model.pth"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, src_vocab, tgt_vocab = load_model_ckpt_q2("q2_model.pth", device)
            st.session_state["model_q2"] = model
            st.session_state["src_vocab_q2"] = src_vocab
            st.session_state["tgt_vocab_q2"] = tgt_vocab
            st.sidebar.success("Loaded saved model.")
        else:
            st.sidebar.error("No saved model found. Please train first.")
    if st.sidebar.button("Train Model (Q2)"):
        st.session_state['cancel_training'] = False
        model, src_vocab, tgt_vocab = train_model(epochs=2, batch_size=4)
        st.session_state["model_q2"] = model
        st.session_state["src_vocab_q2"] = src_vocab
        st.session_state["tgt_vocab_q2"] = tgt_vocab
    st.write("---")
    st.header("Translate C++ to Pseudocode")
    cpp_input = st.text_area("Enter your C++ code:")
    if st.button("Translate", key="q2_translate"):
        if "model_q2" not in st.session_state:
            st.error("No trained model found! Train or load a model first.")
        else:
            with st.spinner("Translating..."):
                pseudo_output = translate_cpp_to_pseudocode(
                    st.session_state["model_q2"],
                    st.session_state["src_vocab_q2"],
                    st.session_state["tgt_vocab_q2"],
                    cpp_input
                )
                st.code(pseudo_output)
if __name__ == "__main__":
    main()
