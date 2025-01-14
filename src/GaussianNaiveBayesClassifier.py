import numpy as np
from .NaiveBayesClassifier import NaiveBayesClassifier
class GaussianNaiveBayesClassifier(NaiveBayesClassifier):
    """
    Klasyfikator Naiwnego Bayesa dla danych ilościowych (ciągłych) z wykorzystaniem rozkładu normalnego.
    
    Atrybuty:
        classes (list): Lista dostępnych klas w danych.
        features (list): Lista cech (kolumn) w danych.
        class_counter (dict): Licznik liczby rekordów dla każdej klasy.
        class_feature_stats (dict): Statystyki (średnia, odchylenie standardowe i wartości) dla każdej cechy w każdej klasie.
        fitted (bool): Flaga wskazująca, czy model został wytrenowany.
    """
    def __init__(self, classes, features):
        """
        Inicjalizuje klasę GaussianNaiveBayesClassifier dziedziczącą po 
        
        Args:
            classes (list): Lista nazw klas (np. ['setosa', 'versicolor', 'virginica']).
            features (list): Lista cech (np. ['sepal length', 'sepal width', 'petal length', 'petal width']).
        """
        super().__init__(classes, features)
        self.class_counter = {c: 0 for c in classes}
        self.class_feature_stats = {c: {f: {'mean': 0, 'std': 0, 'values': []} for f in features} for c in classes}

    def fit(self, data):
        """
        Trenuje model na podanych danych.
        
        Args:
            data (pandas.DataFrame): Dane treningowe zawierające kolumnę 'class' oraz kolumny cech.
        """
        for _, record in data.iterrows():
            c = record.get("class")
            self.class_counter[c] += 1
            for f in self.features:
                if f in record:
                    value = record[f]
                    self.class_feature_stats[c][f]['values'].append(value)

        for c in self.classes:
            for f in self.features:
                self.class_feature_stats[c][f]['mean'] = np.mean(self.class_feature_stats[c][f]['values'])
                self.class_feature_stats[c][f]['std'] = np.std(self.class_feature_stats[c][f]['values'])
        self.fitted = True

    def predict_proba(self, record):
        """
        Oblicza prawdopodobieństwa przynależności rekordu do każdej klasy.
        
        Args:
            record (dict): Słownik reprezentujący pojedynczy rekord (nazwa cechy -> wartość cechy).
        
        Returns:
            probs (dict): Słownik, gdzie kluczem jest nazwa klasy, a wartością obliczone prawdopodobieństwo.
        
        Raises:
            Exception: Jeśli model nie został wytrenowany.
        """
        if not self.fitted:
            raise Exception('Gaussian Naive Bayes Classifier not fitted')
        total_records = sum(self.class_counter.values())
        probs = {c: self.class_counter[c] / total_records for c in self.classes}
        for c in self.classes:
            for f in self.features:
                if f in record:
                    value = record[f]
                    mean = self.class_feature_stats[c][f]['mean']
                    std = self.class_feature_stats[c][f]['std']
                    if std > 0:
                        probs[c] *= self._gauss_value(value, mean, std)
                    else:
                        probs[c] = 0
        return probs                        
    def _gauss_value(self, x, mean, std):
        """
        Funkcja prywatna. Oblicza wartość funkcji gęstości rozkładu normalnego.
        
        Args:
            x (float): Wartość cechy.
            mean (float): Średnia cechy.
            std (float): Odchylenie standardowe cechy.
        
        Returns:
            float: Prawdopodobieństwo uzyskane z funkcji rozkładu Gaussa.
        """
        return np.exp(-((x - mean)**2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
