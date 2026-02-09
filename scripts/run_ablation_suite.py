from __future__ import annotations
import subprocess
import sys

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    py = sys.executable

    # GRU: attention vs no-attention
    run([py, "-m", "scripts.run_train", "--config", "configs/ablations/gru_attention.yaml"])
    run([py, "-m", "scripts.run_train", "--config", "configs/ablations/gru_no_attention.yaml"])

    # dropout=0 vs 0.2
    run([py, "-m", "scripts.run_train", "--config", "configs/ablations/gru_dropout0.yaml"])
    run([py, "-m", "scripts.run_train", "--config", "configs/ablations/gru_dropout02.yaml"])

    # weight_decay=0 vs 1e-4
    run([py, "-m", "scripts.run_train", "--config", "configs/ablations/gru_wd0.yaml"])
    run([py, "-m", "scripts.run_train", "--config", "configs/ablations/gru_wd1e4.yaml"])

    # ensemble K=1 vs K=5
    run([py, "-m", "scripts.run_ensemble", "--config", "configs/gru.yaml", "--k", "1"])
    run([py, "-m", "scripts.run_ensemble", "--config", "configs/gru.yaml", "--k", "5"])

    # naive baselines
    run([py, "-m", "scripts.run_baselines", "--config", "configs/gru.yaml"])

    print("\nDone. Check results/logs/* for metrics JSON.")

if __name__ == "__main__":
    main()
