import matplotlib.pyplot as plt

def plotCollection(ax, ys, *args, **kwargs):

  ax.plot(ys, *args, **kwargs)

  if "label" in kwargs.keys():

    #remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)

    plt.legend(newHandles, newLabels)