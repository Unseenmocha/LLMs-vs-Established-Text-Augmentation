from ClassicalUpsampler import EDA_Upsampler
import pandas as pd

def test_upsample_sentence():
    upsampler = EDA_Upsampler()

    x_train = pd.read_csv('../datasets/spotify/spotify_10_train_unaugmented.csv')
    x_train = x_train['text']
    
    modified_lyrics = upsampler._upsample_sentence(x_train.iloc[3], 2)
    print("Original Lyrics:\n", x_train.iloc[3])
    print("\nModified Lyrics 1:\n", modified_lyrics[0],"\n\nModified Lyrics 2:\n", modified_lyrics[1])

def test_upsample_sentence2():
    upsampler = EDA_Upsampler()

    x_train = pd.read_csv('../datasets/linkedin/linkedin_train.csv')
    x_train = x_train['description']
    
    modified_lyrics = upsampler._upsample_sentence(x_train.iloc[3], 3)
    print("Original desc:\n", x_train.iloc[3])
    print("\nModified desc 1:\n", modified_lyrics[1],"\n\nModified desc 2:\n", modified_lyrics[2])

def test_upsample_dataset():
    upsampler = EDA_Upsampler()
    # Load datasets
    train = pd.read_csv('../datasets/spotify/spotify_10_train_unaugmented.csv')

    train = train.loc[train['label'].isin([0,1,2])].reset_index(drop=True)

    train = train.groupby("label").sample(n=1, random_state=1).reset_index(drop=True)

    print(train)

    x_train_spotify = train[['text']]
    y_train_spotify = train[['label']]


    # Convert to lists
    x_train_spotify = x_train_spotify['text'].tolist()
    y_train_spotify = y_train_spotify['label'].tolist()


    x_train_aug, y_train_aug = upsampler.upsample_dataset(x_train_spotify, y_train_spotify, 2)

    df_train_x = pd.DataFrame(x_train_aug, columns=['text'])
    df_train_y = pd.DataFrame(y_train_aug, columns=['label'])

    df_train = pd.concat([df_train_x, df_train_y], axis=1)
    print(df_train_x.iloc[0,0])
    print('\n\n************************************************************************************\n\n')
    print(df_train_x.iloc[1,0])
    print('\n\n************************************************************************************\n\n')
    print(df_train)


    # Save DataFrames to CSV files
    df_train.to_csv('../datasets/spotify/classical/test_spotify_classical_train.csv', index=False) 


test_upsample_sentence()
# test_upsample_sentence2()
# test_upsample_dataset()