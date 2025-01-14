from .NaiveBayesClassifier import *
class MultinomialNaiveBayesClassifier(NaiveBayesClassifier):
    """
    Klasyfikator Naiwnego Bayesa dla danych kategorycznych (Multinomial).
    
    Atrybuty:
        classes (list): Lista dostępnych klas w danych.
        features (list): Lista cech (kolumn) w danych.
        class_counter (dict): Licznik liczby rekordów dla każdej klasy.
        feature_values (dict): Słownik przechowujący unikalne wartości dla każdej cechy.
        class_feature_value_counts (dict): Liczniki wartości każdej z cech cech dla każdej klasy.
        fitted (bool): Flaga wskazująca, czy model został wytrenowany.
    """
    def __init__(self, classes, features):
        """
        Inicjalizuje klasę MultinomialNaiveBayesClassifier dziedziczącą po 
        
        Args:
            classes (list): Lista nazw klas
            features (list): Lista cech
        """
        super().__init__(classes, features)
        self.class_counter = {c: 0 for c in classes}
        self.feature_values = {f: set() for f in features}
        self.class_feature_value_counts = {c: {f: {} for f in features} for c in classes}

    def fit(self, data):
        """
        Trenuje model na podanych danych.
    
        Args:
            data (pandas.DataFrame): Dane treningowe, zawierające kolumnę 'class' oraz kolumny cech.
        """
        for _, record in data.iterrows():
            c = record.get("class")
            self.class_counter[c] += 1
            for f in self.features:
                if f in record:
                    value = record[f]
                    self.feature_values[f].add(value)
                    if value in self.class_feature_value_counts[c][f]:
                        self.class_feature_value_counts[c][f][value] += 1
                    else:
                        self.class_feature_value_counts[c][f][value] = 1
        self.fitted = True

    def predict_proba(self, record):
        """
        Oblicza prawdopodobieństwa przynależności rekordu do każdej klasy.
        
        Args:
            record (dict): Słownik reprezentujący pojedynczy rekord (nazwa cechy -> wartość cechy).
        
        Returns:
            probs (dict): Słownik, gdzie kluczem jest nazwa klasy (self.classes), a wartością obliczone dla niej prawdopodobieństwo.
        
        Raises:
            Exception: Jeśli model nie został wytrenowany.
        """
        if not self.fitted:
            raise Exception('Multinomial Naive Bayes Classifier not fitted')
        total_records = sum(self.class_counter.values())
        probs = {c: self.class_counter[c] / total_records for c in self.classes}
        for c in self.classes:
            for f in self.features:
                if f in record:
                    value = record[f]
                    count = self.class_feature_value_counts[c][f].get(value, 0)
                    total = sum(self.class_feature_value_counts[c][f].values())
                    if total > 0:
                        probs[c] *= count / total
                    else:
                        probs[c] = 0
        return probs