from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from copy import deepcopy

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

class Bert_Trainer():
    def __init__(self, num_labels, dataset, model_save_name, class_weights=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_labels = num_labels
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.dataset = dataset

        self.class_weights = class_weights

        self.model_save_name=model_save_name
        
        self.train_loader = None
        self.train_dataset = None

        self.test_dataset = None
        self.test_loader = None

    def load_data(self, x_train, y_train, x_test, y_test, x_val, y_val, batch_size=8):
        self.train_dataset = self.dataset(x_train, y_train, self.tokenizer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.test_dataset = self.dataset(x_test, y_test, self.tokenizer)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size)

        self.val_dataset = self.dataset(x_val, y_val, self.tokenizer)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

    def reset_model(self, save_name=None):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels)
        if save_name:
            self.model_save_name = save_name

    def save_model(self, save_name=None):
        torch.save(self.model.state_dict(), f"./models/{save_name if save_name else self.model_save_name}.pth")

    def load_model(self, path):
        state_dict = torch.load('./models/'+path+'.pth')
        self.model.load_state_dict(state_dict)
        

    def train(self, lr=5e-5, epochs=3, validate=True, early_stopping=True, early_stopping_tol=0.02):
        # Define optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=lr)

        # Move the model to the GPU if available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

        loss_fn = nn.CrossEntropyLoss( self.class_weights.to(device) if torch.is_tensor(self.class_weights) else None)

        val_acc = -1
        best_val_acc = -1
        best_state, best_val_loss, best_val_report = None, None, None
        # Train the model
        for epoch in range(epochs):
            print(f"Running Epoch {epoch + 1}/{epochs}...")
            self.model.train()
            state = deepcopy(self.model.state_dict())
            total_loss = 0
            correct_predictions = 0

            for index, batch in enumerate(self.train_loader):
                print(f"computing batch {index}/{len(self.train_loader)}", end='\r')
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct_predictions += (outputs.logits.argmax(dim=1) == labels).sum().item()

            print(f"\nEpoch {epoch + 1}/{epochs} Results                  ")
            print(f"Training loss: {total_loss / len(self.train_loader)}")
            print(f"Training accuracy: {correct_predictions / len(self.train_dataset)}")

            if (validate):
                val_acc, val_loss, val_report = self.evaluate(mode='val')
                print(f"Validation Loss: {val_loss}")
                print(f"Validation Accuracy: {val_acc}\n")

                if (val_acc > best_val_acc):
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_val_report = val_report
                    best_state = deepcopy(self.model.state_dict())

                # early stopping
                if early_stopping and (val_acc < (best_val_acc - early_stopping_tol)):
                    print('stopping early..')
                    self.model.load_state_dict(best_state)
                    return best_val_acc, best_val_loss, best_val_report
                
        self.model.load_state_dict(best_state)
        return best_val_acc, best_val_loss, best_val_report


    def evaluate(self, mode):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Validate the model
        self.model.eval()

        self.model.to(device)

        if (mode == 'test'):
            loader = self.test_loader
            dataset = self.test_dataset
        elif mode == 'val':
            loader = self.val_loader
            dataset = self.val_dataset
        elif mode == 'train':
            loader = self.train_loader
            dataset = self.train_dataset
        else:
            raise ValueError('Invalid mode')

        loss = 0
        correct_predictions = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                preds = outputs.logits.argmax(dim=1)

                loss += outputs.loss.item()
                correct_predictions += (preds == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = correct_predictions / len(dataset)
        loss = loss / len(loader)

        # precision = precision_score(all_labels, all_preds, average='weighted')
        # recall = recall_score(all_labels, all_preds, average='weighted')
        # f1 = f1_score(all_labels, all_preds, average='weighted')
        # print(f"{'Test' if mode == 'test' else 'Validation'} loss: {loss}")
        # print(f"accuracy: {val_accuracy}")
        # print(f"precision: {precision}")
        # print(f"recall: {recall}")
        # print(f"F1 score: {f1}\n\n")

        report = classification_report(all_labels, all_preds)


        return accuracy, loss, report
    