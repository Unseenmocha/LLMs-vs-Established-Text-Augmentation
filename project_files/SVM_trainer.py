import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from DataSets import SVMDataset



class SVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    
class SVM_trainer():
    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val, class_weights=None, batched_vectorization=False, batch_size=1024):
        self.vectorizer = TfidfVectorizer()
        
        self.class_weights = class_weights

        self.vectorizer.fit(x_train)

        self.batch_size = batch_size

        self.batched_vectorization = batched_vectorization
        if batched_vectorization:
            self.x_train = x_train
            size = len(self.vectorizer.transform(x_train[0:5]).toarray()[0])
            self.model = SVM(size, 10)
        else:
            self.x_train = torch.tensor(self.vectorizer.transform(x_train).toarray(), dtype=torch.float)
            self.model = SVM(self.x_train.size()[1],10)


        self.y_train = torch.tensor(y_train)

        self.x_test = torch.tensor(self.vectorizer.transform(x_test).toarray(), dtype=torch.float)
        self.y_test = torch.tensor(y_test)

        self.x_val = torch.tensor(self.vectorizer.transform(x_val).toarray(), dtype=torch.float)
        self.y_val = torch.tensor(y_val)

        
    
    def train(self, lr, weight_decay=None, epochs=100, batched=False, batch_size=1024):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay if weight_decay else 0)
        criterion = nn.MultiMarginLoss(weight = self.class_weights.to(device) if torch.is_tensor(self.class_weights) else None)

        self.model = self.model.to(device)

        if batched:
            dataset = SVMDataset(self.x_train, self.y_train)
            dataloader = DataLoader(dataset, batch_size, True)
            
            for epoch in range(epochs):
                total_predictions = []
                total_labels = []
                for index, batch in enumerate(dataloader):
                    batch_data, batch_labels = batch

                    if self.batched_vectorization:
                        batch_data = torch.tensor(self.vectorizer.transform(list(batch_data)).toarray(), dtype=torch.float)

                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    optimizer.zero_grad()

                    outputs = self.model(batch_data)
                    _, predictions = torch.max(outputs, 1)
                    total_predictions.extend(predictions)
                    total_labels.extend(batch_labels)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                total_predictions = torch.tensor(total_predictions)
                total_labels = torch.tensor(total_labels)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

                    accuracy = (total_predictions == total_labels).float().mean()
                    print(f"Training acc: {accuracy}")

        else:
            

            self.x_train = self.x_train.to(device)
            self.x_test = self.x_test.to(device)
            self.y_train = self.y_train.to(device)
            self.y_test = self.y_test.to(device)

            for epoch in range(epochs):
                
                optimizer.zero_grad()
                outputs = self.model(self.x_train)
                _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs, self.y_train)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

                    accuracy = (predictions == self.y_train).float().mean()
                    print(f"Training acc: {accuracy}")


    def evaluate(self, mode):
        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if mode=='val':
            x = self.x_val
            y = self.y_val
        elif mode == 'test':
            x = self.x_test
            y = self.y_test
        else:
            raise ValueError('Invalid mode')
        
        x = x.to(device)
        y = y.to(device)
        self.model.to(device)
        with torch.no_grad():
            outputs = self.model(x)
            _, predictions = torch.max(outputs, 1)
            accuracy = (predictions == y).float().mean()
            print(f"{'Test' if mode=='test' else 'Validation'} Accuracy: {accuracy.item():.4f}")
            print(classification_report(y.cpu().numpy(), predictions.cpu().numpy()))