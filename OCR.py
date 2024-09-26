import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

class OCRExtractor:
    def __init__(self, dataset_path, model_path='CustomCnn_model.h5', img_shape=(32, 32), batch_size=32):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.mapping = {}
        self.mapping_inverse = {}
        self.model = self.load_model()
        self.prepare_data()

    def directory_to_df(self, path):
        df = []
        chars = 'abcdABCD'    # to include required letters only
        for cls in os.listdir(path):
            cls_path = os.path.join(path, cls)
            cls_name = cls.split('_')[0]
            if cls_name not in chars:
                continue
            for img_path in os.listdir(cls_path):
                direct = os.path.join(cls_path, img_path)
                df.append([direct, cls_name])

        df = pd.DataFrame(df, columns=['image', 'label'])
        print("The number of samples found:", len(df))
        return df.copy()

    def prepare_data(self):
        df = self.directory_to_df(self.dataset_path)
        X, y = df['image'], df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41)
        training_df = pd.concat((X_train, y_train), axis=1)
        X, y = training_df['image'], training_df['label']
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=41)
        training_df = pd.concat((X_train, y_train), axis=1)
        gen = ImageDataGenerator(dtype=np.int32, brightness_range=[0.0, 1.0], fill_mode='nearest')
        train_gen = gen.flow_from_dataframe(training_df, x_col='image', y_col='label', batch_size=self.batch_size, target_size=self.img_shape)
        self.mapping = train_gen.class_indices
        self.mapping_inverse = {v: k for k, v in self.mapping.items()}

    def read_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def convert_2_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def binarization(self, image):
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        return _, thresh

    def dilate(self, image, words=False):
        img = image.copy()
        m = 3
        n = m - 2  # n less than m for Vertical structuring element to dilate chars
        itrs = 4
        if words:
            m = 6
            n = m
            itrs = 3
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, m))
        dilation = cv2.dilate(img, rect_kernel, iterations=itrs)
        return dilation

    def find_rect(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rects = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)  # Extract the bounding rectangle coordinates of each contour
            rects.append([x, y, w, h])

        sorted_rects = list(sorted(rects, key=lambda x: x[0]))  # Sorting the rects from Left-to-Right
        return sorted_rects

    def extract(self, image):
        chars = []  # a list to store recognized characters

        image_cpy = image.copy()
        _, bin_img = self.binarization(self.convert_2_gray(image_cpy))
        full_dil_img = self.dilate(bin_img, words=True)
        words = self.find_rect(full_dil_img)  # Recognized words within the image

        for word in words:
            x, y, w, h = word  # coordinates of the word
            img = image_cpy[y:y + h, x:x + w]

            _, bin_img = self.binarization(self.convert_2_gray(img))
            dil_img = self.dilate(bin_img)
            char_parts = self.find_rect(dil_img)  # Recognized chars within the word
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)  # draw a green rectangle around the word

            for char in char_parts:
                x, y, w, h = char
                ch = img[y:y + h, x:x + w]

                empty_img = np.full((32, 32, 1), 255, dtype=np.uint8)  # a white image used for resize with filling
                x_start, y_start = 3, 3  # starting indices
                resized = cv2.resize(ch, (16, 22), interpolation=cv2.INTER_CUBIC)
                gray = self.convert_2_gray(resized)
                empty_img[y_start:y_start + 22, x_start:x_start + 16, 0] = gray.copy()  # integrate the recognized char into the white image
                gray = cv2.cvtColor(empty_img, cv2.COLOR_GRAY2RGB)
                gray = gray.astype(np.int32)

                predicted = self.mapping_inverse[np.argmax(self.model.predict(np.array([gray]), verbose=0))]
                chars.append(predicted)  # append the character into the list

            chars.append(' ')  # at the end of each iteration (end of word) append a space

        return ''.join(chars[:-1])

    def execute(self):
        ans = []
        for i in range(1, 11):
            img = self.read_image(f'./ocr_slices/img_{i}.jpg')
            text = self.extract(img)
            # print(f"{i} --> {text}")
            ans.append(text)
        return ans
