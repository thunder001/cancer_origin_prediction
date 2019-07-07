
from shutil import copy
from model_training import ModelTraining
from model_testing import run_testing


def run_model_selection(trainfile, testfile,  units, sample_size, model_dir, best_model_dir):

    # get models
    models = {}
    training = ModelTraining()

    for i in range(len(units)):
        print("\nRunning training {0} of {1} ...".format(i + 1, len(units)))
        modelfile = model_dir + "model_" + str(i) + ".ckpt"
        print("Unit of hidden layer: {}".format(units[i]))
        # print(type(units[i]))

        training.run_training(trainfile, units[i], modelfile)

        _, _, _, _, accuracy = run_testing(testfile, modelfile, units[i], sample_size)

        models[units[i]] = accuracy
        print("Accuracy: {}".format(accuracy))

    # choose best models (hidden layer unit)
    max_acc = 0
    for unit, acc in models.items():
        a = models[unit]
        if a > max_acc:
            max_acc = a
    print('Best accuracy is {}'.format(max_acc))
    for unit, acc in models.items():
        if acc == max_acc:
            best_unit = unit
            print('Best hidden units are {}'.format(best_unit))
            best_model_idx = units.index(best_unit)
            best_model_file1 = model_dir + "model_" + str(best_model_idx) + ".ckpt.index"
            best_model_file2 = model_dir + "model_" + str(best_model_idx) + ".ckpt.meta"
            best_model_file3 = model_dir + "model_" + str(best_model_idx) + ".ckpt.data-00000-of-00001"
            copy(best_model_file1, best_model_dir)
            copy(best_model_file2, best_model_dir)
            copy(best_model_file3, best_model_dir)
