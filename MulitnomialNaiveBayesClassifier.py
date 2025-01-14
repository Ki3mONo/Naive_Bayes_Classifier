class MultinomialNaiveBayesClassifier:
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
        self.classes = classes
        self.class_counter = {c: 0 for c in classes}
        self.features = features
        self.feature_values = {f: set() for f in features}
        self.class_feature_value_counts = {c: {f: {} for f in features} for c in classes}
        self.fitted = False

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

    def predict(self, record):
        """
        Przewiduje klasę dla podanego rekordu.
        
        Args:
            record (dict): Słownik reprezentujący pojedynczy rekord (nazwa cechy -> wartość cechy).
        
        Returns:
            pred_class (str): Przewidywana klasa dla rekordu.
        
        Raises:
            Exception: Jeśli model nie został wytrenowany.
        """
        if not self.fitted:
            raise Exception('Multinomial Naive Bayes Classifier not fitted')
        probs = self.predict_proba(record)
        pred_class = max(probs, key=probs.get)
        return pred_class

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
            raise Exception('Multinomial Naive Bayes Classifier not fitted')
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
