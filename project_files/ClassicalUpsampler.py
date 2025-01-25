from DatasetUpsampler import DatasetUpsampler
import numpy as np
import re
import pandas as pd
from textaugment import EDA

import random

def reinsert_newlines_paragraphs(string, new_s):
    # Split original string into paragraphs and lines
    original_paragraphs = string.split('\n\n')
    average_paragraph_length = sum(len(p.split()) for p in original_paragraphs) / len(original_paragraphs)
    
    # Calculate how many double newlines are needed
    augmented_words = new_s.split()
    num_paragraphs_needed = max(1, int(len(augmented_words) / average_paragraph_length))
    
    # Determine evenly spaced double newline positions
    interval = len(augmented_words) / (num_paragraphs_needed + 1)
    double_newline_positions = [
        int(i * interval + random.uniform(-0.1, 0.1) * interval)
        for i in range(1, num_paragraphs_needed)
    ]
    
    double_newline_positions = sorted(set(max(1, min(len(augmented_words) - 1, pos)) for pos in double_newline_positions))
    
    # Insert double newlines first
    for pos in reversed(double_newline_positions):
        augmented_words.insert(pos, '\n\n')
    
    # Handle single newlines within paragraphs
    average_words_per_line = average_paragraph_length / 3  # Assume 3 lines per paragraph on average
    num_lines_needed = max(1, int(len(augmented_words) / average_words_per_line)) - num_paragraphs_needed
    
    single_newline_positions = sorted(
        set(
            int(i * len(augmented_words) / (num_lines_needed + num_paragraphs_needed + 1))
            for i in range(1, num_lines_needed + 1)
        )
    )
    
    # Insert single newlines
    for pos in reversed(single_newline_positions):
        if augmented_words[pos] != '\n\n':
            augmented_words.insert(pos, '\n')
    
    # Rebuild the string with clean formatting
    new_s_formatted = ' '.join(augmented_words).replace(' \n ', '\n').replace(' \n', '\n').replace('\n ', '\n')
    new_s_formatted = new_s_formatted.replace(' \n\n ', '\n\n').replace('\n\n ', '\n\n').replace(' \n\n', '\n\n')
    return new_s_formatted


class EDA_Upsampler(DatasetUpsampler):
    def __init__(self):
        self.t = EDA()

    def _upsample_sentence(self, string, num_target_sentences):
        samples = [string]
        words = re.findall(r"\b\w+(?:'\w+)?\b", string)
        num_words = len(words)
        num_augs = int(0.1*num_words)

        for _ in range(num_target_sentences-1):
            new_s = string
            new_s = self.t.random_insertion(new_s, num_augs)
            new_s = self.t.random_deletion(new_s, 0.1)
            new_s = self.t.random_swap(new_s, num_augs)
            new_s = self.t.synonym_replacement(new_s, num_augs)

            new_s = reinsert_newlines_paragraphs(string, new_s)

            samples.append(new_s)

        return samples


def upsample_spotify():
    upsampler = EDA_Upsampler()

    train = pd.read_csv('../datasets/spotify/spotify_10_train_unaugmented.csv')

     # Convert to lists
    x_train_spotify = train['text'].tolist()
    y_train_spotify = train['label'].tolist()

    x_train_aug, y_train_aug = upsampler.upsample_dataset(x_train_spotify, y_train_spotify, 1000)

    df_train_x = pd.DataFrame(x_train_aug, columns=['text'])
    df_train_y = pd.DataFrame(y_train_aug, columns=['label'])

    df_train = pd.concat([df_train_x, df_train_y], axis=1)

    print(df_train_x.iloc[0,0])
    print('\n\n************************************************************************************\n\n')
    print(df_train_x.iloc[1,0])
    print('\n\n************************************************************************************\n\n')
    print(df_train)

    df_train.to_csv('../datasets/spotify/classical/spotify_classical_train.csv', index=False) 

def upsample_linkedin():
    upsampler = EDA_Upsampler()
    # Load datasets
    linkedin_data = pd.read_csv('../datasets/linkedin/linkedin_train.csv')
    # titles = linkedin_data['title'].drop_duplicates().reset_index(drop=True)


    # title_to_idx = {title: idx for idx, title in enumerate(titles)}
    # linkedin_data['title_numerical'] = linkedin_data['title'].map(title_to_idx)

    x = linkedin_data['description'].tolist()
    y = linkedin_data['title'].tolist()

    x_train_aug, y_train_aug = upsampler.upsample_dataset(x, y, 1000)

    df_train_x = pd.DataFrame(x_train_aug, columns=['description'])
    df_train_y = pd.DataFrame(y_train_aug, columns=['title'])

    df_train = pd.concat([df_train_x, df_train_y], axis=1)

    print(df_train_x.iloc[0,0])
    print('\n\n************************************************************************************\n\n')
    print(df_train_x.iloc[1,0])
    print('\n\n************************************************************************************\n\n')
    print(df_train)

    df_train.to_csv('../datasets/linkedin/classical/linkedin_classical_train.csv', index=False)  


# test_upsample_dataset()

# test_upsample_sentence2()

upsample_spotify()
upsample_linkedin()