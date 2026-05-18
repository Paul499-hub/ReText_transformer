import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

def make_test_wav(path: str = "test_tone.wav"):
    sample_rate = 24_000
    seconds = 3
    t = torch.arange(sample_rate * seconds) / sample_rate
    wav = 0.2 * torch.sin(2 * torch.pi * 440 * t)
    wav = wav.unsqueeze(0)  # [channels, samples]
    torchaudio.save(
        str(path),
        wav,
        sample_rate,
        format="wav",
        backend="soundfile",
    )
    print(f"saved: {path}")

def tokenize_audio(path: str, model:EncodecModel, prints:bool=False):
    path = str(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, sample_rate = torchaudio.load(path)
    model.set_target_bandwidth(6.0)
    model.to(device)
    model.eval()
    wav = convert_audio(
        wav,
        sample_rate,
        model.sample_rate,
        model.channels,
    )
    wav = wav.unsqueeze(0).to(device)  # [batch, channels, samples]
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
    if prints:
        print(f"audio path: {path}")
        print(f"waveform shape: {wav.shape}")
        print(f"token codes shape: {codes.shape}")
        print(f"token dtype: {codes.dtype}")
        print(f"first tokens: {codes[0, :, :10].cpu()}")
    return codes
