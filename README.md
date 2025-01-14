# Naive Bayes Classifier

## Opis projektu
Projekt implementuje Naiwne Klasyfikatory Bayesowskie dla dwóch typów danych:
1. **Dane kategoryczne** – implementacja klasyfikatora **Multinomial Naive Bayes Classifier**.
2. **Dane ilościowe** – implementacja klasyfikatora **Gaussian Naive Bayes Classifier**.

Projekt testowano na rzeczywistych zbiorach danych, takich jak zbiór Iris dostępny w bibliotece scikit-learn i zbiór Mushroom dostępny na platformie Kaggle (katalog /data).

---

## Klasyfikatory
### 1. Multinomial Naive Bayes Classifier
- Przeznaczony dla danych kategorycznych.
- Oblicza prawdopodobieństwo na podstawie liczby wystąpień wartości cech dla każdej klasy.
- Zakłada warunkową niezależność cech.

### 2. Gaussian Naive Bayes Classifier
- Przeznaczony dla danych ilościowych.
- Opiera się na rozkładzie normalnym, obliczając prawdopodobieństwa na podstawie średniej i odchylenia standardowego cech.

---

## Struktura projektu
1. **Implementacja klasyfikatorów: /src**
   - `MultinomialNaiveBayesClassifier` – dla danych kategorycznych.
   - `GaussianNaiveBayesClassifier` – dla danych ilościowych.
   - `NaiveBayesClassifier` – abstrakcyjna klasa nadrzędna dla naiwnego klasyfikatora bayesowskiego.

2. **Wstępna analiza danych: /data_analysis**
   - Wizualizacja danych.
   - Wybór cech, które najlepiej odróżniają klasy w zbiorze danych.

3. **Ewaluacja: /accuracy_notebooks**
   - Podział danych na zbiory treningowe i testowe.
   - Obliczanie dokładności poszczególnych klasyfikatorów w notebookach `Mushrooms_accuracy.ipynb` i `Iris_Accuracy.ipynb`.

---

## Technologie
- **Python** – główny język programowania.
- **Biblioteki:**
  - `numpy` – obliczenia matematyczne.
  - `pandas` – manipulacja danymi.
  - `scikit-learn` – wczytywanie zbiorów danych i podział na zbiory treningowe/testowe.

---

## Autorzy
- **Maciej Kmąk**
- **Jakub Gucwa**

Projekt został napisany jako projekt z przedmiotu **Rachunek prawdopodobieństwa i statystyka** na II roku studiów **Informatyka** na Wydziale Informatyki AGH.
