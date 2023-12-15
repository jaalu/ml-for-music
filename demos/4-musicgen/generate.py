import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(1)    # generates an unconditional audio sample
audio_write(f'unconditional', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

description = 'happy rock'
wav = model.generate([description]) 
audio_write(f'{description}', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

