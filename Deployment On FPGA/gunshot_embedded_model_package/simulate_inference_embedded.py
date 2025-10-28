
import torch, numpy as np, os, pandas as pd
import matplotlib.pyplot as plt
from model_architecture import BiLSTM_Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_Attention().to(device)
model.load_state_dict(torch.load("gunshot_embedded_model.pth", map_location=device))
model.eval()

def predict_embedding(np_file):
    x = np.load(np_file)
    if x.ndim == 1:
        x = x.reshape(1, 1, -1)
    else:
        x = x.reshape(1, *x.shape)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
    label = "🔫 Gunshot" if pred == 1 else "🌳 Non-Gunshot"
    conf = probs[pred]
    print(f"🎯 {os.path.basename(np_file)} → {label} ({conf*100:.2f}%)")
    return os.path.basename(np_file), label, conf

def predict_folder(folder_path):
    results = []
    for f in os.listdir(folder_path):
        if f.endswith('.npy'):
            results.append(predict_embedding(os.path.join(folder_path, f)))
    df = pd.DataFrame(results, columns=["File", "Prediction", "Confidence"])
    csv_path = os.path.join(folder_path, "inference_results_embedded.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to {csv_path}")
    plt.figure(figsize=(10,5))
    plt.barh(df["File"], df["Confidence"], color=["#4CAF50" if p=="🌳 Non-Gunshot" else "#E74C3C" for p in df["Prediction"]])
    plt.xlabel("Confidence")
    plt.title("Gunshot Detection on Embeddings")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gunshot Detection using Embeddings-only Model")
    parser.add_argument("--file", type=str, help="Path to a single .npy embedding file")
    parser.add_argument("--folder", type=str, help="Path to a folder of .npy embeddings")
    args = parser.parse_args()

    if args.file:
        predict_embedding(args.file)
    elif args.folder:
        predict_folder(args.folder)
    else:
        print("⚠️ Please provide --file or --folder")
