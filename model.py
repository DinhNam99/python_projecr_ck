import argparse
import cv2
import imghdr
import numpy as np
import os
import pathlib

from halo import Halo
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

class Model:
    def __init__(self, data_dir_path: str = '', skip_evaluation: bool = False):
        self.skip_evaluation = skip_evaluation
        self.data_dir_path = data_dir_path

        if not(self.data_dir_path and self.data_dir_path.strip()):
            self.data_dir_path = 'data/'

        models_dir_path = os.path.abspath('models/')
        pathlib.Path(models_dir_path).mkdir(parents=True, exist_ok=True)

        self.joblib_path = os.path.join(models_dir_path, 'classifier.joblib')

    def predict(self, img):
        """
        Dự đoán ảnh
        """

        clf = load(self.joblib_path)

        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img = img.flatten()

        label = clf.predict([img])[0]
        return label

    def read_dataset(self):
        """
        Phương pháp đọc tập dữ liệu hình ảnh trong mảng numpy để training mô hình.
        Cấu trúc của thư mục dữ liệu tại `data_dir_path` phải là:
            data
            ├── a
            |   ├── 1.png
            |   ├── 2.png
            |   └── ...
            ├── b
            |   ├── 1.png
            |   ├── 2.png
            |   └── ...
            └── ...
        Tên thư mục con là tên của nhãn và chứa hình ảnh viết tay tuân thủ định dạng MNIST của ký tự được biểu thị bởi nhãn đó.
        """

        base_sep_count = self.data_dir_path.count(os.sep)
        features = []
        labels = []

        spinner = Halo(text='Đọc...', spinner='dots')
        spinner.start()

        for subdir, dirs, files in os.walk(self.data_dir_path, topdown=True):
            if subdir.count(os.sep) - base_sep_count == 1:
                del dirs[:]
                continue

            for filename in files:
                filepath = os.path.join(subdir, filename)

                if imghdr.what(filepath) == 'png':
                    labels.append([os.path.basename(os.path.dirname(filepath))])

                    name_im = cv2.imread(filename=filepath)
                    name_im = np.dot(name_im[..., :3], [0.299, 0.587, 0.114])
                    name_im = name_im.flatten()

                    features.append(list(name_im))

        spinner.succeed(text='Đọc hoàn thành...')

        features = np.array(features)
        labels = np.array(labels).ravel()

        return features, labels

    def train(self):
        """
        Phương thức traning
        """

        # Bước 1 : đọc data
        x, y = self.read_dataset()

        # Chia ra train và test
        features_train, features_test, labels_train, labels_test = train_test_split(x, y, test_size=0.2, random_state=0)

        spinner = Halo(text='Trainning...', spinner='dots')
        spinner.start()

        # step 2: Đào tạo mô hình
        pipeline_knn_clf = Pipeline([('scaler', MinMaxScaler()), ('classifier', KNeighborsClassifier())])
        pipeline_knn_clf.fit(features_train, labels_train)

        spinner.succeed(text='Hoàn thành...')

        if self.skip_evaluation:
            print('Bỏ qua đánh giá chạy')

        else:
            spinner = Halo(text='Đánh giá...', spinner='dots')
            spinner.start()

            # step 3: Đánh giá mô hình
            labels_pred = pipeline_knn_clf.predict(features_test)
            score = accuracy_score(labels_test, labels_pred)

            spinner.succeed(text='Đánh giá hoàn thành')

            print(f'Độ chính xác (max=1.0): {score}')


        dump(pipeline_knn_clf, self.joblib_path)

        print(f'Lưu mô hình tại {self.joblib_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
            Trainning mô hình Phân loại kNN.
            """,
        usage='%(prog)s [options]',
    )
    parser.add_argument(
        '-dap',
        '--data-path',
        dest='data_path',
        type=str,
        help='Đường dẫn nơi lưu trữ hình ảnh dữ liệu.',
    )
    parser.add_argument(
        '-se',
        '--skip-evaluation',
        dest='skip_evaluation',
        action='store_true',
        default=False,
        help='Có nên bỏ qua các đánh giá đang chạy hay không.',
    )
    args = parser.parse_args()

    model = Model(data_dir_path=args.data_path, skip_evaluation=args.skip_evaluation)
    model.train()
