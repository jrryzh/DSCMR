import numpy as np
import matplotlib as plt
from matplotlib import cm
import pylab

hsv_colors = [(0.56823266219239377, 0.82777777777777772, 0.70588235294117652),
              (0.078146611341632088, 0.94509803921568625, 1.0),
              (0.33333333333333331, 0.72499999999999998, 0.62745098039215685),
              (0.99904761904761907, 0.81775700934579443, 0.83921568627450982),
              (0.75387596899224807, 0.45502645502645506, 0.74117647058823533),
              (0.028205128205128216, 0.4642857142857143, 0.5490196078431373),
              (0.8842592592592593, 0.47577092511013214, 0.8901960784313725),
              (0.0, 0.0, 0.49803921568627452),
              (0.16774193548387095, 0.82010582010582012, 0.74117647058823533),
              (0.51539855072463769, 0.88888888888888884, 0.81176470588235294)]

rgb_colors = plt.colors.hsv_to_rgb(np.array(hsv_colors).reshape(10, 1, 3))
colors = plt.colors.ListedColormap(rgb_colors.reshape(10, 3))


def plot(Y, labels):
    pylab.scatter(Y[:, 0], Y[:, 1], s=30, c=labels, cmap=cm.get_cmap('tab20'), linewidth=0)
    pylab.show()


if __name__ == '__main__':
    x = np.random.normal(2, 5, 20)
    y = np.random.normal(3, 3, 20)
    pylab.scatter(x, y, s=20, c=list(range(0, 20)), cmap=cm.get_cmap('tab20'))
    pylab.legend(loc='upper left')
    pylab.show()
