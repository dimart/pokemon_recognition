from os import listdir, makedirs
from os.path import join, exists, isdir
from time import time

import numpy as np

from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as pl
from scipy.misc import imread
from PIL import Image, ImageOps
from bunch import Bunch

POKEMON_PATH = './pokemon'
POKEMON_PROC_PATH = './pokemon_processed'


# original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())


# original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def get_pokemon(h=200, w=200):
    # processing data: make all images h X w in size and gray scale, save them in diff folder
    if not exists(POKEMON_PROC_PATH):
        makedirs(POKEMON_PROC_PATH)

        for pokemon_name in sorted(listdir(POKEMON_PATH)):
            folder = join(POKEMON_PATH, pokemon_name)
            if not isdir(folder):
                continue

            # make new dir for the pokemon
            new_folder = join(POKEMON_PROC_PATH, pokemon_name)
            makedirs(new_folder)

            # iterate over existing pokemon's pictures and process each one
            paths = [join(folder, f) for f in listdir(folder) if f != '.DS_Store']
            for i, path in enumerate(paths):
                img = Image.open(path).convert('RGB')
                img = ImageOps.fit(img, (w, h), Image.ANTIALIAS, (0.5, 0.5))
                img = ImageOps.grayscale(img)

                new_path = join(POKEMON_PROC_PATH, pokemon_name, str(i)) + '.jpg'
                img.save(new_path)

    # read pokemon names and paths
    pokemon_names, pokemon_paths = [], []
    for pokemon_name in sorted(listdir(POKEMON_PROC_PATH)):
        folder = join(POKEMON_PROC_PATH, pokemon_name)
        if not isdir(folder):
            continue
        paths = [join(folder, f) for f in listdir(folder) if f != '.DS_Store']
        n_images = len(paths)
        pokemon_names.extend([pokemon_name] * n_images)
        pokemon_paths.extend(paths)

    n_pokemon = len(pokemon_paths)
    target_names = np.unique(pokemon_names)
    target = np.searchsorted(target_names, pokemon_names)

    # read data
    pokemons = np.zeros((n_pokemon, h, w), dtype=np.float32)
    for i, pokemon_path in enumerate(pokemon_paths):
            img = imread(pokemon_path)
            pokemon = np.asarray(img, dtype=np.uint32)
            pokemons[i, ...] = pokemon

    # shuffle pokemon
    indices = np.arange(n_pokemon)
    np.random.RandomState(42).shuffle(indices)
    pokemons, target = pokemons[indices], target[indices]

    return Bunch(data=pokemons.reshape(len(pokemons), -1), images=pokemons,
                 target=target, target_names=target_names,
                 DESCR="Pokemon dataset")


def main():
    np.random.seed(3)
    pokemon = get_pokemon()

    h, w = pokemon.images[0].shape

    X = pokemon.data
    y = pokemon.target
    n_classes = pokemon.target_names.shape[0]

    kf = KFold(len(y), n_folds=4, shuffle=True)
    scores = 0.0
    t0 = time()
    for train_index, test_index in kf:
        X_train = np.array([X[i] for i in train_index])
        X_test = np.array([X[i] for i in test_index])
        y_train = np.array([y[i] for i in train_index])
        y_test = np.array([y[i] for i in test_index])

        ###############################################################################
        # Apply PCA
        n_components = 18  # 80%
        pca = PCA(n_components=n_components, whiten=True).fit(X_train)
        eigenfaces = pca.components_.reshape((n_components, h, w))

        print "Projecting the input data on the eigenpokemon orthonormal basis"
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # reconstruction = pca.inverse_transform(X_train_pca[1])
        # im = Image.fromarray(reconstruction.reshape(h,w))
        # im.show()

        ###############################################################################
        # Train a SVM classification model
        print "Fitting the classifier to the training set"
        param_grid = {
                'kernel': ['rbf', 'linear'],
                'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        }
        clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
        clf = clf.fit(X_train_pca, y_train)

        # print "Fitting the classifier to the training set"
        # param_grid = {
        #         'n_neighbors': list(xrange(1, 15)),
        # }
        # clf = GridSearchCV(KNeighborsClassifier(), param_grid)
        # clf = clf.fit(X_train, y_train)

        ###############################################################################
        # Quantitative evaluation of the model quality on the test set
        print "Predicting pokemon names on the testing set"
        y_pred = clf.predict(X_test_pca)

        print classification_report(y_test, y_pred, target_names=pokemon.target_names)
        print confusion_matrix(y_test, y_pred, labels=range(n_classes))
        scores += clf.score(X_test_pca, y_test)

        ###############################################################################
        # View results
        prediction_titles = [title(y_pred, y_test, pokemon.target_names, i)
                             for i in range(y_pred.shape[0])]
        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]

        plot_gallery(X_test, prediction_titles, h, w)
        plot_gallery(eigenfaces, eigenface_titles, h, w)
        pl.show()

    print "Computed in %0.3fs" % (time() - t0)
    print 'AVG score = %0.3f' % (scores/len(kf))

if __name__ == "__main__":
    main()