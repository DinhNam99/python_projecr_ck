import cv2
import numpy as np

from model import Model

class Drawer:
    def __init__(self):
        self.mouse_pressed = False
        self.img = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8)
        self.char_color = (255, 255, 255)

    def draw(self):
        """
            Phương thức vẽ các chữ cái cần nhận dạng.
        """
        self.reset()

        window_name = 'Viết chữ'

        cv2.namedWindow(winname=window_name)
        cv2.setMouseCallback(window_name=window_name, on_mouse=self.mouse_callback)
        while True:
            cv2.imshow(winname=window_name, mat=self.img)

            # Nhấn Esc để thoát.
            k = cv2.waitKey(delay=1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

    def get_contours(self):
        """
        Phương thức tìm đường viền trong ảnh và cắt chúng và trả về danh sách có đường viền được cắt
        """

        images = []
        main_image = self.img
        orig_image = main_image.copy()

        # Chuyển đổi sang thang độ xám và áp dụng lọc Gaussian
        main_image = cv2.cvtColor(src=main_image, code=cv2.COLOR_BGR2GRAY)
        main_image = cv2.GaussianBlur(src=main_image, ksize=(5, 5), sigmaX=0)

        _, main_image = cv2.threshold(src=main_image, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

        # Tìm đường viên các ảnh con
        contours, _ = cv2.findContours(image=main_image.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        # Lấy hình chữ nhật chứa mỗi đường viền
        bboxes = [cv2.boundingRect(array=contour) for contour in contours]

        for bbox in bboxes:
            x, y, width, height = bbox[:4]
            images.append(orig_image[y:y + height, x:x + width])

        return images

    def get_images(self):
        images = []

        self.draw()

        char_images = self.get_contours()

        for cimg in char_images:
            images.append(Drawer.convert_to_emnist(img=cimg))

        return images

    def mouse_callback(self, event, x, y, flags, params):
        """
        Phương thức gọi lại để vẽ vòng tròn trên một hình ảnh
        """

        # Nhấn đúp chuột
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True

        # Di chuyển con trỏ chuột
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_pressed:
                cv2.circle(img=self.img, center=(x, y), radius=20, color=self.char_color, thickness=-1)

        # Nhấn chuột trái
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
            cv2.circle(img=self.img, center=(x, y), radius=20, color=self.char_color, thickness=-1)

    def reset(self):
        # Đặt lại
        self.img = np.zeros((1024, 1024, 3), np.uint8)

    @staticmethod
    def convert_to_emnist(img):
        """
        Phương thức để làm cho một hình ảnh định dạng EMNIST tương thích dịnh dạng .img
        """

        height, width = img.shape[:2]

        # Tạo một khung hình vuông có chiều dài bằng kích thước lớn nhất
        emnist_image = np.zeros(shape=(max(height, width), max(height, width), 3), dtype=np.uint8)

        # Cắt trung tâm ảnh
        offset_height = int(float(emnist_image.shape[0] / 2.0) - float(height / 2.0))
        offset_width = int(float(emnist_image.shape[1] / 2.0) - float(width / 2.0))

        emnist_image[offset_height:offset_height + height, offset_width:offset_width + width] = img

        # thay đổi kích thước thành 26x26
        emnist_image = cv2.resize(src=emnist_image, dsize=(26, 26), interpolation=cv2.INTER_CUBIC)

        # chỉnh lại 26x26 đến 28x28
        fin_image = np.zeros(shape=(28, 28, 3), dtype=np.uint8)
        fin_image[1:27, 1:27] = emnist_image

        return fin_image


if __name__ == '__main__':
    images = Drawer().get_images()
    labels = []
    for image in images:
        label = Model().predict(img=image)
        labels.append(label)
    labels.reverse()
    print("".join(labels) )