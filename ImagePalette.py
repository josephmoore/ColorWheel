import numpy as np
import cv2
from sklearn.cluster import KMeans

#TODO fix channel sorting in get_palette_image

class ImagePalette():

    def __init__(self):
        pass

    def get_palette(self, image_n, n_clusters):
        """load image and return palette of n_clusters"""
        image = cv2.cvtColor((cv2.imread(image_n)), cv2.COLOR_BGR2HLS)
        #reshape image to list of pixels
        self.img = image.reshape((image.shape[0] * image.shape[1], 3))
        self.clt = KMeans(n_clusters)
        self.clt.fit(self.img)
        palette = self.clt.cluster_centers_

        return palette
    
    def save_palette_image(self, dst_image): 
        self.clt.fit(self.img)
        hist = self._centroid_histogram(self.clt)
        bar = self._plot_colors(hist, self.clt.cluster_centers_)
        bar = cv2.cvtColor(bar, cv2.COLOR_HLS2BGR) #I don't know why it's not 2RGB

        cv2.imwrite(dst_image, bar)

    def hls2rgb(self, hls_colors):
        """takes a list of HLS color components
            and return RGV equivalent."""
        rgb_colors = []
        for color in hls_colors:
            color8 = color.astype(np.uint8)
            rgb_color = cv2.cvtColor(np.array([[[c for c in color8]]]), cv2.COLOR_HLS2RGB)
            #unpack rgb values from muli-dimension array
            rgb_color = rgb_color.tolist()[0][0]
            rgb_colors.append(rgb_color)

        return rgb_colors

    def _plot_colors(self, hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype = "uint8")
        startX = 0

        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            #print(color.astype("uint8").tolist())
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
              color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar

    def _centroid_histogram(self, clt):
          # grab the number of different clusters and create a histogram
          # based on the number of pixels assigned to each cluster
          numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
          (hist, _) = np.histogram(clt.labels_, bins = numLabels)
          # normalize the histogram, such that it sums to one
          hist = hist.astype("float")
          hist /= hist.sum()
          # return the histogram
          return hist

