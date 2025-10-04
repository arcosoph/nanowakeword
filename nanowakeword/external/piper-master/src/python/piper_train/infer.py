# #!/usr/bin/env python3
# import argparse
# import json
# import logging
# import sys
# import time
# from pathlib import Path

# import torch

# from .vits.lightning import VitsModel
# from .vits.utils import audio_float_to_int16
# from .vits.wavfile import write as write_wav

# _LOGGER = logging.getLogger("piper_train.infer")


# def main():
#     """Main entry point"""
#     logging.basicConfig(level=logging.DEBUG)
#     parser = argparse.ArgumentParser(prog="piper_train.infer")
#     parser.add_argument(
#         "--checkpoint", required=True, help="Path to model checkpoint (.ckpt)"
#     )
#     parser.add_argument("--output-dir", required=True, help="Path to write WAV files")
#     parser.add_argument("--sample-rate", type=int, default=22050)
#     #
#     parser.add_argument("--noise-scale", type=float, default=0.667)
#     parser.add_argument("--length-scale", type=float, default=1.0)
#     parser.add_argument("--noise-w", type=float, default=0.8)
#     #
#     args = parser.parse_args()

#     args.output_dir = Path(args.output_dir)
#     args.output_dir.mkdir(parents=True, exist_ok=True)

#     model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)

#     # Inference only
#     model.eval()

#     with torch.no_grad():
#         model.model_g.dec.remove_weight_norm()

#     for i, line in enumerate(sys.stdin):
#         line = line.strip()
#         if not line:
#             continue

#         utt = json.loads(line)
#         utt_id = str(i)
#         phoneme_ids = utt["phoneme_ids"]
#         speaker_id = utt.get("speaker_id")

#         text = torch.LongTensor(phoneme_ids).unsqueeze(0)
#         text_lengths = torch.LongTensor([len(phoneme_ids)])
#         scales = [args.noise_scale, args.length_scale, args.noise_w]
#         sid = torch.LongTensor([speaker_id]) if speaker_id is not None else None

#         start_time = time.perf_counter()
#         audio = model(text, text_lengths, scales, sid=sid).detach().numpy()
#         audio = audio_float_to_int16(audio)
#         end_time = time.perf_counter()

#         audio_duration_sec = audio.shape[-1] / args.sample_rate
#         infer_sec = end_time - start_time
#         real_time_factor = (
#             infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
#         )

#         _LOGGER.debug(
#             "Real-time factor for %s: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
#             i + 1,
#             real_time_factor,
#             infer_sec,
#             audio_duration_sec,
#         )

#         output_path = args.output_dir / f"{utt_id}.wav"
#         write_wav(str(output_path), args.sample_rate, audio)


# if __name__ == "__main__":
#     main()



# piper_train/infer.py er sheshe add korun
import torch
from .vits.lightning import VitsModel
from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav

class Piper:
    def __init__(self, checkpoint, device="cpu", sample_rate=22050):
        self.device = device
        self.sample_rate = sample_rate
        self.model = VitsModel.load_from_checkpoint(checkpoint, dataset=None)
        self.model.eval()
        with torch.no_grad():
            self.model.model_g.dec.remove_weight_norm()
    
    @staticmethod
    def load(checkpoint, device="cpu"):
        return Piper(checkpoint, device=device)
    
    def infer_batch(self, texts, noise_scale=0.667, length_scale=1.0, noise_w=0.8, speaker_ids=None):
        """
        texts: list of phoneme id lists
        speaker_ids: list of speaker IDs (or None)
        returns: list of np arrays (audio)
        """
        import numpy as np

        audios = []
        for i, phoneme_ids in enumerate(texts):
            text = torch.LongTensor(phoneme_ids).unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(phoneme_ids)]).to(self.device)
            sid = (
                torch.LongTensor([speaker_ids[i]]).to(self.device)
                if speaker_ids is not None
                else None
            )
            scales = [noise_scale, length_scale, noise_w]
            with torch.no_grad():
                audio = self.model(text, text_lengths, scales, sid=sid).detach().cpu().numpy()
            audio = audio_float_to_int16(audio)
            audios.append(audio)
        return audios

    def save_audio(self, audio, out_path):
        write_wav(out_path, self.sample_rate, audio)

