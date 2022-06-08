import cv2 as cv
import numpy as np
import tkinter as tk
import pytesseract as tess
from matplotlib import pyplot as plt
from tkinter import filedialog, messagebox
tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('600x450')

        # Add a menubar
        self.main_menu = tk.Menu(window)

        # Add file submenu
        self.file_menu = tk.Menu(self.main_menu, tearoff=0)
        self.file_menu.add_command(label='開啟檔案', command=self.open_file)
        self.file_menu.add_command(label='儲存檔案', command=self.save_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label='離開程式', command=window.quit)

        # Add operation submenu
        self.operation_menu = tk.Menu(self.main_menu, tearoff=0)
        self.operation_menu.add_command(label='圖片資訊', command=self.info)
        self.operation_menu.add_command(label='灰階轉換', command=self.gray)
        self.operation_menu.add_command(label='色彩空間', command=self.image_contrast_enhance)
        self.operation_menu.add_command(label='RGB直方圖', command=self.show_color_histogram)
        # self.operation_menu.add_command(label='直方圖均衡化', command=self.opencv_histogram_equalization)
        self.operation_menu.add_command(label='遮罩', command=self.show_histogram_with_subplot)
        # self.operation_menu.add_command(label='剪裁', command=self.cut)
        self.operation_menu.add_separator()
        self.operation_menu.add_command(label='縮放', command=self.resize)
        self.operation_menu.add_command(label='平移', command=self.translate)
        self.operation_menu.add_command(label='旋轉', command=self.rotate)
        self.operation_menu.add_command(label='仿射', command=self.affine)

        # Add filter submenu
        self.filter_menu = tk.Menu(self.main_menu, tearoff=0)
        self.filter_menu.add_command(label='平滑化模糊', command=self.averaging_filter)
        self.filter_menu.add_command(label='高斯模糊', command=self.gaussian_filter)
        self.filter_menu.add_command(label='中值濾波', command=self.median_filter)
        self.filter_menu.add_command(label='索伯算子', command=self.sobel_filter)
        self.filter_menu.add_command(label='拉普拉斯子', command=self.laplacian_filter)
        self.filter_menu.add_command(label='高增幅', command=self.unsharp_mask)
        self.filter_menu.add_command(label='景深', command=self.bilateral_filter)
        # self.filter_menu.add_command(label='Canny Edge', command=self.canny_edge)

        # Add ocr submenu
        self.ocr_menu = tk.Menu(self.main_menu, tearoff=0)
        self.ocr_menu.add_command(label='文字辨識', command=self.tesseract_ocr)

        # Add submenu to mainmenu
        self.main_menu.add_cascade(label='檔案', menu=self.file_menu)
        self.main_menu.add_cascade(label='功能', menu=self.operation_menu)
        self.main_menu.add_cascade(label='濾鏡', menu=self.filter_menu)
        self.main_menu.add_cascade(label='辨識', menu=self.ocr_menu)

        # display menu
        self.window.config(menu=self.main_menu, cursor='circle')

        # # add a video source
        # self.video_source = video_source
        # # open video source
        # self.vid = MyVideoCapture(self.video_source)
        # # create a canvas to display the video content
        # self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        # self.canvas.pack()
        # # create a button to capture the frame
        # self.btn_snapshot = tk.Button(window, text='snapshot', width='50', command=self.snapshot)
        # self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        # self.delay = 15
        # self.update()
        self.window.mainloop()

    def open_file(self):
        filetypes = (
            ('jpg files', '*.jpg'),
            ('png files', '*.png'),
            ('All files', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes
        )
        global img
        img = cv.imread(filename)
        cv.imshow(filename, img)

    def save_file(self):
        filetypes = (
            ('jpg files', '*.jpg'),
            ('png files', '*.png'),
            ('All files', '*.*')
        )
        filename = filedialog.asksaveasfilename(
            title='Save a file',
            initialdir='/',
            filetypes=filetypes
        )
        cv.imwrite(filename, img)

    def info(self):
        img_info = img.shape
        tk.messagebox.showinfo(title='image info', message=img_info)

    def gray(self):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('gray', img_gray)

    # 調亮或調暗
    def image_contrast_enhance(self):
        new_image = np.zeros(img.shape, img.dtype)
        gamma = 5
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    new_image[y, x, c] = np.clip(pow(img[y, x, c] / 255.0, gamma) * 255.0, 0, 255)
        cv.imshow('Contrast enhanced', new_image)

    # RGB分布
    def show_color_histogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()

    # 灰階直方圖
    def show_gray_histogram(self):
        cv.imshow('original gray image', self)
        plt.hist(self.ravel(), 256, [0, 256])
        plt.show()
        pass

    # 直方圖等化(均衡化)
    def opencv_histogram_equalization(self):
        plt.figure(1)
        plt.hist(img.ravel(), 256, [0, 256])
        img_eq = cv.equalizeHist(img)
        cv.imshow('equalized image', img_eq)
        plt.hist(img_eq.ravel(), 256, [0, 256])
        plt.show()

    # 遮罩
    def show_histogram_with_subplot(self):
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[100:300, 200:500] = 255  # 遮罩範圍
        masked_image = cv.bitwise_and(img, img, mask=mask)
        cv.imshow('masked image', masked_image)
        hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
        hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])
        plt.subplot(2, 2, 1)  # 圖表組合
        plt.imshow(img, 'gray')
        plt.subplot(2, 2, 2)
        plt.imshow(mask, 'gray')
        plt.subplot(2, 2, 3)
        plt.imshow(masked_image, 'gray')
        plt.subplot(2, 2, 4)
        plt.plot(hist_full)
        plt.plot(hist_mask)
        plt.xlim([0, 256])
        plt.show()

    # 剪取範圍 (ROI)
    def click_and_crop(self, event, x, y, flags, param):
        clone = img.copy()
        global refPt, cropping
        refPt = []
        cropping = False
        if event == cv.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
        elif event == cv.EVENT_LBUTTONUP:
            refPt.append((x, y))
            cropping = False
            cv.rectangle(clone, refPt[0], refPt[1], (255, 255, 255), 1)
            cv.imshow('image', clone)

    def cut(self):
        clone = img.copy()
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.click_and_crop)
        while True:
            cv.imshow('image', img)
            key = cv.waitKey(1) & 0xFF
            if key == ord('r'):
                cv.imshow('image', clone)
                # img = clone.copy()
            elif key == ord('c'):
                break
        if len(refPt) == 2:
            roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            cv.imshow('ROI', roi)

    # 縮放
    def resize(self):
        rows, cols, ch = img.shape
        img_res = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
        cv.imshow('resize image', img_res)

    # 平移
    def translate(self):
        rows, cols, ch = img.shape
        M = np.float32([[1, 0, 100],
                        [0, 1, 50]])
        img_tra = cv.warpAffine(img, M, (cols, rows))
        cv.imshow('translate image', img_tra)

    # 旋轉
    def rotate(self):
        rows, cols, ch = img.shape
        M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -45, 1)  # 中心點,角度,倍率
        img_rot = cv.warpAffine(img, M, (cols, rows))
        cv.imshow('rotate image', img_rot)

    # 仿射
    def affine(self):
        rows, cols, ch = img.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 80], [200, 50], [100, 250]])
        M = cv.getAffineTransform(pts1, pts2)
        img_aff = cv.warpAffine(img, M, (cols, rows))
        cv.imshow('affine image', img_aff)

    # 平滑化
    def averaging_filter(self):
        # averaging blur filter
        img_averaging = cv.blur(img, (11, 11))
        cv.imshow('blur image', img_averaging)

    # 高斯模糊
    def gaussian_filter(self):
        # Gaussian blur filter
        img_gaussian_blur = cv.GaussianBlur(img, (11, 11), -1)
        cv.imshow('Gaussian blur image', img_gaussian_blur)

    # 中值濾波
    def median_filter(self):
        # median filter
        img_median = cv.medianBlur(img, 11)
        cv.imshow('median blur image', img_median)

    # 索伯算子
    def sobel_filter(self):
        # Sobel filter
        x = cv.Sobel(img, cv.CV_16S, 1, 0)
        y = cv.Sobel(img, cv.CV_16S, 0, 1)
        abs_x = cv.convertScaleAbs(x)  # 轉回uint8
        abs_y = cv.convertScaleAbs(y)
        img_sobel = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        # cv.imshow('x-direction gradient image', abs_x)
        # cv.imshow('y-direction gradient image', abs_y)
        cv.imshow('Sobel image', img_sobel)

    # 拉普拉斯子
    def laplacian_filter(self):
        # Laplacian filter
        gray_lap = cv.Laplacian(img, cv.CV_16S, ksize=3)
        img_laplacian = cv.convertScaleAbs(gray_lap)  # 轉回uint8
        cv.imshow('Laplacian image', img_laplacian)

    # 高增幅影像
    def unsharp_mask(self):
        kernel_size = (5, 5)
        amount = 1.5
        img_blur = cv.GaussianBlur(img, kernel_size, 1.0)
        # cv.imshow('blurred image', img_blur)
        # cv.imshow('sharpen image', img - img_blur)
        img_usm = cv.addWeighted(img, amount + 1.0, img_blur, -1.0 * amount, 0)
        img_usm = np.clip(img_usm, 0, 255)
        cv.imshow('unsharp image', img_usm)

    # 景深(雙值濾波)
    def bilateral_filter(self):
        img_bi = cv.bilateralFilter(img, 9, 100, 100)  # 參數調整
        cv.imshow('bilateral image', img_bi)

    def canny_edge(self):
        pass

    def tesseract_ocr(self):
        image = img
        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        text = tess.image_to_string(img_rgb, lang="eng")
        h, w, c = image.shape
        boxes = tess.image_to_boxes(image)
        for b in boxes.splitlines():
            b = b.split(' ')
            image = cv.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

        cv.imshow('text detect', image)
        tk.messagebox.showinfo(title='文字辨識結果', message=text)

    # def update(self):
    #     # get a frame from the video source
    #     ret, frame = self.vid.get_frame()
    #     if ret:
    #         self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    #         self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    #     self.window.after(self.delay, self.update)

    # def snapshot(self):
    #     # get a frame from video source
    #     ret, frame = self.vid.get_frame()
    #     if ret:
    #         cv.imwrite('test.jpg', cv.cvtColor(frame, cv.COLOR_RGB2BGR))

    # image_contrast_enhance(img_ori)
    # image_contrast_enhance(img_gray)
    # show_color_histogram(img_ori)
    # show_gray_histogram(img_gray)
    # show_histogram_with_subplot(img_gray)
    # opencv_histogram_equalization(img_gray)

    # cv.waitKey(0)
    # cv.destroyAllWindows()


App(tk.Tk(), 'OpenCV with Tkinter GUI')
