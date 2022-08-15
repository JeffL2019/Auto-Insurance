import numpy as np
import matplotlib.pyplot as plt


def acc():
    data = {'Decision Tree':0.9168571428571428, 'Random Forest':0.9192857142857143, 'Gradient Boosting':0.8507142857142858,
            'Ada':0.843, 'Neural Network':0.8542857142857143}
    model = list(data.keys())
    acc = list(data.values())

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(model, acc)

    plt.xlabel("Model")
    plt.ylabel("Accuracy Rating")
    plt.title("Accuracy Rating By Model")
    plt.show()

def runtime():
    data = {
        'Decision Tree': 0.01914525032043457,
        'Random Forest': 0.5165872573852539,
        'Gradient Boosting': 0.42074108123779297,
        'Ada': 0.4326636791229248,
        'Neural Network': 4.370111465454102
    }
    model = list(data.keys())
    rt = list(data.values())

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(model, rt)

    plt.xlabel("Model")
    plt.ylabel("Runtime")
    plt.title("Runtime By Model")
    plt.show()

acc()
# runtime()
