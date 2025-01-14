from abc import ABC, abstractmethod 
class NaiveBayesClassifier(ABC):
    """
    Klasa nadrzędna dla Naiwnych Klasyfikatorów Bayesowskich.
    
    Atrybuty:
        classes (list): Lista dostępnych klas w danych.
        features (list): Lista cech (kolumn) w danych.
        fitted (bool): Flaga wskazująca, czy model został wytrenowany.
    """
    def __init__(self, classes, features):
        """
        Inicjalizuje klasę nadrzędną NaiveBayesClassifier.
        
        Args:
            classes (list): Lista nazw klas.
            features (list): Lista cech.
        """
        self.classes = classes
        self.features = features
        self.fitted = False

    @abstractmethod
    def fit(self, data):
        """
        Metoda Abstrakcyjna
        Trenuje model na podanych danych.
        
        Args:
            data (pandas.DataFrame): Dane treningowe.
        """
        raise NotImplementedError("Metoda fit musi być zaimplementowana w klasie potomnej.")

    @abstractmethod
    def predict_proba(self, record):
        """
        Metoda Abstrakcyjna
        Oblicza prawdopodobieństwa przynależności rekordu do każdej klasy.
        
        Args:
            record (dict): Pojedynczy rekord danych.
        
        Returns:
            dict: Prawdopodobieństwa dla każdej klasy.
        """
        raise NotImplementedError("Metoda predict_proba musi być zaimplementowana w klasie potomnej.")

    def predict(self, record):
        """
        Przewiduje klasę dla podanego rekordu.
        
        Args:
            record (dict): Słownik reprezentujący pojedynczy rekord (nazwa cechy -> wartość cechy).
        
        Returns:
            str: Przewidywana klasa dla rekordu.
        
        Raises:
            Exception: Jeśli model nie został wytrenowany.
        """
        if not self.fitted:
            raise Exception('Naive Bayes Classifier not fitted')
        probs = self.predict_proba(record)
        return max(probs, key=probs.get)

    def predict_dataframe_with_accuracy(self, data):
        """
        Oblicza predykcje dla zbioru danych oraz dokładność modelu.
        
        Args:
            data (pandas.DataFrame): Dane testowe zawierające kolumnę 'class' i kolumny cech.
        
        Returns:
            tuple: 
                - predictions (list): Lista przewidywanych klas dla każdego rekordu.
                - accuracy (float): Dokładność modelu (proporcja poprawnych predykcji).
        
        Raises:
            Exception: Jeśli model nie został wytrenowany.
        """
        if not self.fitted:
            raise Exception('Naive Bayes Classifier not fitted')
        predictions = []
        correct = 0
        for _, row in data.iterrows():
            actual_class = row['class']
            record = row.to_dict()
            pred_class = self.predict(record)
            predictions.append(pred_class)
            if pred_class == actual_class:
                correct += 1
        accuracy = correct / len(data)
        return predictions, accuracy