from ImagePalette import ImagePalette
import numpy as np
import cv2
import json
from multiprocessing import Pool
import glob
import os
from sklearn.cluster import KMeans
import shutil
import time

class ImageSorter():

    def __init__(self):
        """src_dir allows images to be moved at a later point"""
        self.images_data = []
        self.n_clusters = 4

    def create_images_data(self, images_n):
        for img_n in images_n:
            base_name = os.path.basename(img_n)
            dir_name = os.path.dirname(img_n) + "/"
            self.images_data.append({'name' : base_name, 'src_dir' : dir_name})

    def add_size_data(self):
        """adds width, heigh data to images_list"""
        for img_d in self.images_data:
            img = cv2.imread(img_d['src_dir'] + img_d['name'])
            height, width = img.shape[:2]
            img_d['width'] = width
            img_d['height'] = height

    def get_palette(self, img_d):
        """use multiprocessing to add data to many images, not this method"""
        IP = ImagePalette()
        img_d['colors'] = IP.get_palette(img_d['src_dir'] + img_d['name'],self.n_clusters).tolist()
        #IP.save_palette_image(img_d['src_dir'] + img_d['name'] + ".jpg")
        return img_d

    def sort_by_width(self):
        def get_width(img_d):
            return img_d['width']

        self.images_data = sorted(self.images_data, key=get_width, reverse=True)

    def sort_by_hue(self, c_index):
        """uses the hue of the first color in the colors list"""
        def get_hue(img_d):
            return img_d['colors'][c_index]

        self.images_data = sorted(self.images_data, key=get_hue, reverse=True)

    def sort_images_saturation(self):
        """sorts the image saturation of each image in images_data"""
        def get_sat(colors):
            return colors[2]

        for img_d in self.images_data:
            img_d['colors'] = sorted(img_d['colors'], key=get_sat, reverse=True)

    def save_images_data(self, out_json):
        with open(out_json, 'w') as f:
            json.dump(self.images_data, f, indent=2)

    def write_reordered_images(self, src_dir, dst_dir):
        """changes the name of images in self.images_data based on
            their current order, e.g. my_image.jpg becomes 0000_my_image.jpg"""
        i=0
        padd_width = len(str(len(self.images_data)))
        for img_d in self.images_data:
            new_name = dst_dir + str(i).zfill(padd_width) + "_" + img_d['name']
            shutil.copy(src_dir + img_d['name'], new_name)
            i=i+1

    
#using separate functions to add color data as 
#methods won't allow pickling and therefore  multiprocessing
def get_palette(image_d):
    image_n = image_d['src_dir'] + image_d['name']
    image = cv2.cvtColor((cv2.imread(image_n)), cv2.COLOR_BGR2HLS)
    #reshape image to list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(3) #3-4 color clusters seems to work best
    clt.fit(image)
    palette = clt.cluster_centers_
    image_d['colors'] = palette.tolist()
    return image_d
    
def get_palette_data():
    pool = Pool(processes=4)
    new_images_list = pool.map(get_palette, IS.images_data)
    pool.close()
    pool.join()
    return new_images_list

if __name__ == "__main__":
    src_dir = "cam_images/640x480_100x/"
    dst_dir = "cam_images/640x480_reordered/"
    images_n = glob.glob(src_dir + "*.jpg")    
    #start = time.time()
    IS = ImageSorter()
    IS.create_images_data(images_n) #set up inital data: image name and src_dir
    IS.add_size_data()
    new_images_data = get_palette_data()
    IS.images_data = new_images_data
    #stop = time.time()
    IS.sort_images_saturation() #make the most saturated color the first in the list
    IS.sort_by_hue(0)  
    IS.save_images_data(src_dir + "image_data.json")
    src_dir = "cam_images/tested_jpgs_640x480/"
    IS.write_reordered_images(src_dir, dst_dir)
    #print(stop - start)

