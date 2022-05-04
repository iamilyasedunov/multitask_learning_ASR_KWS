import sys
sys.path.append(sys.path[0] + "/..")

from utils.utils import *

class AugsCreation():

    def __init__(self):
        self.background_noises = [
                'speech_commands/_background_noise_/white_noise.wav',
                'speech_commands/_background_noise_/dude_miaowing.wav',
                'speech_commands/_background_noise_/doing_the_dishes.wav',
                'speech_commands/_background_noise_/exercise_bike.wav',
                'speech_commands/_background_noise_/pink_noise.wav',
                'speech_commands/_background_noise_/running_tap.wav'
        ]


    def add_rand_noise(self, audio):
        
        # randomly choose noise
        noise_num = torch.randint(low=0, high=len(self.background_noises), size=(1,)).item()    
        noise = torchaudio.load(self.background_noises[noise_num])[0].squeeze()    
        
        noise_level = torch.Tensor([1])  # [0, 40]

        noise_energy = torch.norm(noise)
        audio_energy = torch.norm(audio)
        alpha = (audio_energy / noise_energy) * torch.pow(10, -noise_level / 20)

        start = torch.randint(low=0, high=int(noise.size(0) - audio.size(0) - 1), size=(1,)).item()
        noise_sample = noise[start : start + audio.size(0)]

        audio_new = audio + alpha * noise_sample
        audio_new.clamp_(-1, 1)
        return audio_new


    def __call__(self, wav):
        aug_num = torch.randint(low=0, high=4, size=(1,)).item()   # choose 1 random aug from augs
        augs = [
            lambda x: x,
            lambda x: (x + distributions.Normal(0, 0.01).sample(x.size())).clamp_(-1, 1),
            lambda x: torchaudio.transforms.Vol(.25)(x),
            lambda x: self.add_rand_noise(x)
        ]
        
        return augs[aug_num](wav)
