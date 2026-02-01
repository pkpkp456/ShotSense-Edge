
import torch, librosa, numpy as np, os, pandas as pd, matplotlib.pyplot as plt
import IPython.display as ipd
from model_architecture import GunshotModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GunshotModel(cnn14=None).to(device)
model.load_state_dict(torch.load("gunshot_model_final.pth", map_location=device))
model.eval()

def preprocess_audio(file_path, sr=16000, duration=2.0):
    y, _ = librosa.load(file_path, sr=sr)
    if len(y) < int(sr * duration):
        y = np.pad(y, (0, int(sr * duration) - len(y)))
    else:
        y = y[:int(sr * duration)]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_db = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float().to(device)
    return y, mel_db, sr, mel

def predict_visualize(file_path, show_plot=True):
    y, mel_db, sr, mel = preprocess_audio(file_path)
    with torch.no_grad():
        out = model(mel_db)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
    label = "🔫 Gunshot" if pred == 1 else "🌳 Non-Gunshot"
    conf = probs[pred]
    print(f"🎯 Prediction: {label} | Confidence: {conf:.2f}")
    ipd.display(ipd.Audio(y, rate=sr))
    if show_plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.title(f"{label} ({conf*100:.1f}% Confidence)")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()
    return file_path, label, conf

def predict_folder(folder_path):
    results = []
    for f in os.listdir(folder_path):
        if f.lower().endswith('.wav'):
            path = os.path.join(folder_path, f)
            res = predict_visualize(path, show_plot=False)
            results.append(res)
    df = pd.DataFrame(results, columns=["File", "Prediction", "Confidence"])
    csv_path = os.path.join(folder_path, "inference_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to {csv_path}")

    plt.figure(figsize=(10, 5))
    plt.barh(df["File"], df["Confidence"], color=["#4CAF50" if p=="🌳 Non-Gunshot" else "#E74C3C" for p in df["Prediction"]])
    plt.xlabel("Confidence")
    plt.title("Gunshot Detection Results")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gunshot Detection Inference & Visualization")
    parser.add_argument("--file", type=str, help="Path to a single .wav file")
    parser.add_argument("--folder", type=str, help="Path to a folder of .wav files")
    args = parser.parse_args()
    if args.file:
        predict_visualize(args.file)
    elif args.folder:
        predict_folder(args.folder)
    else:
        print("⚠️ Please provide either --file or --folder argument.")
