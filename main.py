from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os


def main():
    print("---- Train file ----")
    train = get_files_and_language("train/")

    print("\n---- Test file ----")
    test = get_files_and_language("test/", train.keys())

    train = load_data(train)
    test = load_data(test)

    list_predict, score = prediction(train, test)
    list_predict = list(list_predict)

    print(f"\n ========= Prediction score : {str(score*100)}% ========= ")
    for predict, real in zip(list_predict, list(test.keys())):
        check_prediction = "good prediction" if predict == real else "bad prediction"
        print(f"- Language predict: {predict} / real: {real} ====> {check_prediction}")

    print("Process End")


def prediction(train, test):
    clf = Pipeline([
        ('vec', CountVectorizer(analyzer='word')),
        ('clf', MultinomialNB()),
    ])

    clf.fit(list(train.values()), list(train.keys()))

    predict = clf.predict(test.values())
    score = clf.score(list(test.values()), list(test.keys()))

    return predict, score


def load_data(files):
    data = {}
    for lang, path in files.items():
        with open(path, 'r') as f:
            data[lang] = f.read()
            f.close()
    return data


def get_files_and_language(dir_path, possible_choice=""):
    files = {}
    for file in os.listdir(dir_path):
        lang = prompt(f"What is the language of '{file}' file ? ({file}) ", file, possible_choice)
        files[lang] = dir_path + file
    return files


def prompt(text, default, possible_choice):
    inp = input(text)
    while inp not in possible_choice and possible_choice:
        print("Unknown language")
        inp = input(text)
    if not inp:
        inp = default
    return inp


if __name__ == '__main__':
    main()
