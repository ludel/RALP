from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os


def main():
    print("---- Train file ----")
    train = get_all_files("train/")

    print("\n---- Test file ----")
    test = get_all_files("test/")

    train = load_data(train)
    test = load_data(test)

    list_predict = list(prediction(train, test))

    print("\n ========= Prediction =========")
    for predict, real in zip(list_predict, list(test.keys())):
        check_prediction = "good prediction" if predict == real else "bad prediction"
        print(f"- Language predict: {predict} / real: {real} ====> ", check_prediction)

    print("Process End")


def prediction(train, test):
    clf = Pipeline([
        ('vec', TfidfVectorizer(analyzer='word', use_idf=False)),
        ('clf', MultinomialNB()),
    ])

    clf.fit(list(train.values()), list(train.keys()))

    return clf.predict(test.values())


def load_data(files):
    data = {}
    for lang, path in files.items():
        with open(path, 'r') as f:
            data[lang] = f.read()
            f.close()
    return data


def get_all_files(dir_path):
    files = {}
    for file in os.listdir(dir_path):
        lang = prompt(f"What is the language of '{file}' file ? ({file})", file)
        files[lang] = dir_path + file
    return files


def prompt(text, default):
    inp = input(text)
    if not inp:
        inp = default
    return inp


if __name__ == '__main__':
    main()
