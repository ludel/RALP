from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd


def main():
    files = {'python': 'train/python',
             'ruby': 'train/ruby',
             'html': 'train/html',
             'css': 'train/css',
             'js': 'train/js',
             'java': 'train/java',
             'go': 'train/go'}

    data = load_data(files)

    clf = Pipeline([
        ('vec', TfidfVectorizer(analyzer='word', use_idf=False)),
        ('clf', MultinomialNB()),
    ])

    clf.fit(list(data.values()), list(data.keys()))

    test = load_data({"python": "test/python_test",
                      "css": "test/css_test",
                      "html": "test/html_test",
                      "java": 'test/java_test',
                      "go": 'test/go_test'})

    list_predict = list(clf.predict(test.values()))

    for predict, real in zip(list_predict, list(test.keys())):
        print(f"- Language predict {predict} / real {real}")


def load_data(files):
    data = {}
    for lang, path in files.items():
        with open(path, 'r') as f:
            data[lang] = f.read()
            f.close()
    return data


if __name__ == '__main__':
    main()
