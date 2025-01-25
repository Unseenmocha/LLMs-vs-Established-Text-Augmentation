import pandas as pd
import torch

class Linkedin_Data_Reader():
    def __init__(self):
        self.test = pd.read_csv('./datasets/linkedin/linkedin_test.csv')
        self.val = pd.read_csv('./datasets/linkedin/linkedin_val.csv')

        self.train = pd.read_csv('./datasets/linkedin/linkedin_train.csv')
        self.train_backtrans = pd.read_csv('./datasets/linkedin/backtrans/linkedin_train_backtranslated_augmented.csv')
        self.train_llm = pd.read_csv('./datasets/linkedin/llm/linkedin_train_llm_augmented.csv')
        self.train_classical = pd.read_csv('./datasets/linkedin/classical/linkedin_classical_train.csv')

        titles = self.train['title'].unique()
        title_to_idx = {title: idx for idx, title in enumerate(titles)}
        self.title_to_idx = title_to_idx
        self.map_to_ints = lambda set: list(map(lambda title: title_to_idx[title], set))
    
    def read_training(self):
        x_train = self.train['description'].tolist()
        y_train = self.train['title'].tolist()

        x_train_backtrans = self.train_backtrans['description'].tolist()
        y_train_backtrans = self.train_backtrans['title'].tolist()

        x_train_llm = self.train_llm['description'].tolist()
        y_train_llm = self.train_llm['title'].tolist()

        x_train_classical = self.train_classical['description'].tolist()
        y_train_classical = self.train_classical['title'].tolist()

    
        y_train = self.map_to_ints(y_train)
        y_train_backtrans = self.map_to_ints(y_train_backtrans)
        y_train_llm = self.map_to_ints(y_train_llm)
        y_train_classical = self.map_to_ints(y_train_classical)

        return x_train, y_train, x_train_backtrans, y_train_backtrans, x_train_llm, y_train_llm, x_train_classical, y_train_classical

    def get_num_labels(self):
        return len(self.val['title'].drop_duplicates().reset_index(drop=True))
    
    def get_class_weights(self, mode='noaug'):
        if mode == 'noaug':
            set = self.train
        elif mode == 'backtrans':
            set = self.train_backtrans
        elif mode == 'llm':
            set = self.train_llm
        elif mode =='classical':
            set = self.train_classical
        else:
            raise ValueError('invalid mode')

        class_weights = set.groupby('title').count()

        class_weights = (1/class_weights['description'])
        class_weights = torch.tensor(class_weights/class_weights.sum(), dtype=torch.float32)
        return class_weights

    def read_test_val(self):        
        x_test = self.test['description'].tolist()
        y_test = self.test['title'].tolist()
        
        x_val = self.val['description'].tolist()
        y_val = self.val['title'].tolist()


        y_test = self.map_to_ints(y_test)
        y_val = self.map_to_ints(y_val)

        return x_test, y_test, x_val, y_val


class Spotify_Data_Reader():
    def __init__(self):
        self.train = pd.read_csv('./datasets/spotify/spotify_10_train_unaugmented.csv')
        self.train_backtrans = pd.read_csv('./datasets/spotify/backtrans/spotify_10_train_augmented.csv')
        self.train_classical = pd.read_csv('./datasets/spotify/classical/spotify_classical_train.csv')
        self.train_llm = pd.read_csv('./datasets/spotify/llm/spotify_10_train_llm_augmented.csv')
        self.test = pd.read_csv('./datasets/spotify/spotify_10_test.csv')
        self.val = pd.read_csv('./datasets/spotify/spotify_10_val.csv')

    def read_training(self):
        x_train = self.train['text'].tolist()
        y_train = self.train['label'].tolist()
        x_train_backtrans = self.train_backtrans['text'].tolist()
        y_train_backtrans = self.train_backtrans['label'].tolist()
        x_train_classical = self.train_classical['text'].tolist()
        y_train_classical = self.train_classical['label'].tolist()
        x_train_llm = self.train_llm['text'].tolist()
        y_train_llm = self.train_llm['label'].tolist()

        return x_train, y_train, x_train_backtrans, y_train_backtrans, x_train_classical, y_train_classical, x_train_llm, y_train_llm
    
    def get_num_labels(self):
        return len(self.val['label'].drop_duplicates().reset_index(drop=True))
    
    def get_class_weights(self, mode='noaug'):
        if mode == 'noaug':
            set = self.train
        elif mode == 'backtrans':
            set = self.train_backtrans
        elif mode == 'llm':
            set = self.train_llm
        elif mode == 'classical':
            set = self.train_classical
        else:
            raise ValueError('invalid mode')

        class_weights = set.groupby('label').count()

        class_weights = (1/class_weights['text'])
        class_weights = torch.tensor(class_weights/class_weights.sum(), dtype=torch.float32)
        return class_weights

    def read_test_val(self):
        x_test = self.test['text'].tolist()
        y_test = self.test['label'].tolist()

        x_val = self.val['text'].tolist()
        y_val = self.val['label'].tolist()

        return x_test, y_test, x_val, y_val