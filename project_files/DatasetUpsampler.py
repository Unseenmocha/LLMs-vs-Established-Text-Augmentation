import random
from typing import List, Tuple, Any
from collections import defaultdict

class DatasetUpsampler:
    def __init__(self):
        pass

    def _upsample_sentence(self, string: str, num_target_sentences: int) -> List[str]:
        """
        Generate an upsampled list of a given sentence.

        Args:
            string (str): The input sentence to be upsampled.
            num_target_sentences (int): Target number of sentences to generate.

        Returns:
            List[str]: A list containing the upsampled sentences.
        """
        pass  # Define the sentence upsampling logic here

    def _upsample_class(self, string_arr: List[str], num_target_sentences: int) -> List[str]:
        """
        Upsample a list of sentences to reach a target number of sentences for a specific class.

        Args:
            string_arr (List[str]): The list of sentences belonging to a class.
            num_target_sentences (int): Target number of sentences for the class.

        Returns:
            List[str]: A list containing upsampled sentences for the class.
        """
        if num_target_sentences <= len(string_arr):
            return string_arr
        else:
            # Calculate how many sentences are needed per original sentence in string_arr
            sentences_per_string = num_target_sentences // len(string_arr)
            remaining_sentences = num_target_sentences % len(string_arr)
            
            # Upsample each sentence and aggregate results
            upsampled_sentences = []
            for i, string in enumerate(string_arr):
                # Upsample with an extra sentence if needed to reach exact count
                target_count = sentences_per_string + (1 if i < remaining_sentences else 0)
                upsampled_sentences.extend(self._upsample_sentence(string, target_count))
            
            assert len(upsampled_sentences) == num_target_sentences

            return upsampled_sentences

    def upsample_dataset(self, X: List[str], y: List[str], min_sentences_per_class: int) -> Tuple[List[str], List[str]]:
        """
        Upsample a dataset to ensure each class has a minimum number of sentences.

        Args:
            X (List[Any]): The list of data items.
            y (List[Any]): The list of labels corresponding to each data item in X.
            min_sentences_per_class (int): Minimum number of sentences per class after upsampling.

        Returns:
            Tuple[List[Any], List[Any]]: Upsampled data (X) and corresponding labels (y).
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if min_sentences_per_class <= 0:
            raise ValueError("min_sentences_per_class must be a positive integer.")

        # Group X by labels in y
        X_by_y = defaultdict(list)
        for x_item, y_label in zip(X, y):
            X_by_y[y_label].append(x_item)

        # Upsample each class to the minimum required sentences
        for y_label, X_arr in X_by_y.items():
            X_by_y[y_label] = self._upsample_class(X_arr, min_sentences_per_class)

        # Combine upsampled data and labels
        X_upsampled = []
        y_upsampled = []
        for y_label, X_arr in X_by_y.items():
            X_upsampled.extend(X_arr)
            y_upsampled.extend([y_label] * len(X_arr))

        # Randomize X_upsampled and y_upsampled
        combined = list(zip(X_upsampled, y_upsampled))
        random.shuffle(combined)
        X_upsampled, y_upsampled = zip(*combined)

        return list(X_upsampled), list(y_upsampled)

