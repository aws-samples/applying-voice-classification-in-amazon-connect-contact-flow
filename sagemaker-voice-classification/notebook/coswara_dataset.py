import torchaudio
import torch
from pathlib import Path
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

status2idx = {
    "healthy": 0,
    "resp_illness_not_identified": 1,
    "no_resp_illness_exposed": 1,
    "recovered_full": 1,
    "positive_mild": 1,
    "positive_asymp": 1,
    "positive_moderate": 1,
}


class CoswareDataset(Dataset):
    def __init__(self, csv_path: Path, file_path: Path, new_sr=8000, audio_len=20, sampling_ratio=5):
        """Create cosware dataset. Assume single channel 0.

        Args:
            csv_path (Path): Path to dataset metadata csv
            file_path (Path): Path to data folders. Prefix can be found in metadata csv
            new_sr (int, optional): New sampling rate. Defaults to 8000.
            audio_len (int, optional): Audio length based on new sampling rate (sec). Defaults to 20.
            sampling_ratio (int, optional): Additional downsampling ratio. Defaults to 5.
        """

        df = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []
        self.new_sr = new_sr
        self.audio_len = audio_len
        self.sampling_ratio = sampling_ratio
        self.channel = 0

        for i in range(len(df)):
            # Get file fullpath
            if int(df.iloc[i,1])>1024:      ## small file may cause RuntimeError or ExecuteUserScriptError in __getitem__ function
                file_names = file_path / df.iloc[i, 0]
                label = status2idx[df.iloc[i, 2]]
                self.file_names.append(file_names)
                self.labels.append(label)

    def __getitem__(self, index):
        self.curfile = self.file_names[index]
        try:
            waveform, sample_rate = torchaudio.load(self.curfile)
            waveform = torchaudio.transforms.Resample(sample_rate, self.new_sr)(waveform[self.channel, :].view(1, -1))

            # Zero padding to keep desired audio length in seconds
            const_len = self.new_sr * self.audio_len
            tempData = torch.zeros([1, const_len])
            if waveform.shape[1] < const_len:
                tempData[0, : waveform.shape[1]] = waveform[:]
            else:
                tempData[0, :] = waveform[0, :const_len]
            sound = tempData

            # Resampling, need to cater for audio shorter than new length
            const_len = self.new_sr * self.audio_len
            tempData = torch.zeros([1, const_len])
            if sound.shape[1] < const_len:
                tempData[0, : sound.shape[1]] = sound[:]
            else:
                tempData[0, :] = sound[0, :const_len]
            sound = tempData
            # Resampling
            new_const_len = const_len // self.sampling_ratio
            soundFormatted = torch.zeros([1, new_const_len])
            soundFormatted[0, :] = sound[0, ::5]
            
            return soundFormatted, self.labels[index]
        
        except RuntimeError:
            print("Get Item RuntimeError for file: {}".format(self.curfile))
        except ExecuteUserScriptError: 
            print("Get Item ExecuteUserScriptError for file: {}".format(self.curfile))

    def __len__(self):
        return len(self.file_names)

    def plot_soundwave(self):
        """Plot tensor."""
        # Torch is channel first, need to transpose back to numpy
        x = self.soundFormatted.t().numpy()
        plt.figure(figsize=(15, 5))
        plt.plot(x)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    csv_path = Path("/home/ec2-user/efs_project/coswara/data/breathing-deep-metadata.csv")
    file_path = Path("/home/ec2-user/efs_project/coswara/data")

    dataset = CoswareDataset(
        csv_path=csv_path,
        file_path=file_path,
        new_sr=8000,
        audio_len=20,
        sampling_ratio=5,
    )
    # Check 1st samples
    data, label = dataset[0]
    print(label)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    print("Done")