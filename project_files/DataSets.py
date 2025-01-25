from torch.utils.data import Dataset
import torch

# Tokenize lyrics and prepare the dataset
class LyricsDataset(Dataset):
    def __init__(self, lyrics, labels, tokenizer, max_len=512):
        self.lyrics = lyrics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyric = self.lyrics[idx]
        label = self.labels[idx]

        encodings = self.tokenizer(lyric,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=self.max_len,
                                   return_tensors='pt')

        input_ids = encodings['input_ids'].squeeze()  # remove the extra dimension
        attention_mask = encodings['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


class JobDescDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_len=512):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        label = self.labels[idx]

        encodings = self.tokenizer(description,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=self.max_len,
                                   return_tensors='pt')

        input_ids = encodings['input_ids'].squeeze()  # remove the extra dimension
        attention_mask = encodings['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SVMDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label