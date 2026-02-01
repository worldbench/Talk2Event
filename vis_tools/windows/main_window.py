import os
import numpy as np
import pickle
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QTextEdit,\
                            QLabel, QLineEdit, QListWidget
from ..utils import gl_engine
from ..utils import image_utils

class CustomListWidget(QListWidget):

    deleteKeyPressed = pyqtSignal()
    upKeyPressed = pyqtSignal()
    downKeyPressed = pyqtSignal()

    def __init__(self, parent=None):
        super(CustomListWidget, self).__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.deleteKeyPressed.emit()
        elif event.key() == Qt.Key_Up:
            self.upKeyPressed.emit()
        elif event.key() == Qt.Key_Down:
            self.downKeyPressed.emit()
        else:
            super(CustomListWidget, self).keyPressEvent(event)

class MainWindow(QWidget):

    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.setWindowTitle("Window")
        self.setGeometry(100, 100, 800, 600)
        self.sample_index = 0
        self.logger = gl_engine.create_logger()
        self.init_window()
    
    def init_window(self):
        main_layout = QHBoxLayout()
        self.init_dataload_window()
        self.init_display_window()
        self.init_functions_window()
        main_layout.addLayout(self.dataload_layout)
        main_layout.addLayout(self.display_layout)
        main_layout.addLayout(self.functions_layout)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 6)
        main_layout.setStretch(2, 2)
        self.setLayout(main_layout)
        self.current_pixmap = None

    def init_dataload_window(self):
        self.dataload_layout = QVBoxLayout()
        self.results_pkl_select_cbox = QComboBox()
        self.results_pkl_select_cbox.activated.connect(self.results_pkl_selected)
        self.dataload_layout.addWidget(self.results_pkl_select_cbox)
        all_files = os.listdir('/data/yyang/workspace/magiclidar/exps/record_results')
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        self.results_pkl_select_cbox.addItems(pkl_files)

        self.frame_list = CustomListWidget()
        self.frame_list.itemClicked.connect(self.on_frame_selected)
        self.dataload_layout.addWidget(self.frame_list)

        self.save_sample_button = QPushButton('Save sample')
        self.dataload_layout.addWidget(self.save_sample_button)
        self.save_sample_button.clicked.connect(self.save_sample)


    def init_display_window(self):
        self.display_layout = QVBoxLayout()
        temp_layout = QHBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.event_label = QLabel(self)
        self.event_label.setAlignment(Qt.AlignCenter)
        temp_layout.addWidget(self.image_label)
        temp_layout.addWidget(self.event_label)
        self.display_layout.addLayout(temp_layout)
        # "Qlabel"
        h_layout = QHBoxLayout()
        self.sample_index_info = QLabel("")
        self.sample_index_info.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.sample_index_info)
        self.goto_sample_index_box = QLineEdit(self)
        h_layout.addWidget(self.goto_sample_index_box)
        self.goto_sample_index_box.setPlaceholderText('')

        self.goto_sample_index_button = QPushButton('GoTo')
        h_layout.addWidget(self.goto_sample_index_button)
        self.goto_sample_index_button.clicked.connect(self.goto_sample_index)

        self.display_layout.addLayout(h_layout)
        h_layout = QHBoxLayout()

        # << button
        self.prev_view_button = QPushButton('<<<')
        h_layout.addWidget(self.prev_view_button)
        self.prev_view_button.clicked.connect(self.decrement_index)
        # >> button
        self.next_view_button = QPushButton('>>>')
        self.next_view_button.clicked.connect(self.increment_index)
        h_layout.addWidget(self.next_view_button)
        self.display_layout.addLayout(h_layout)

        self.display_layout.setStretch(0, 8)
        self.display_layout.setStretch(1, 1)
        self.display_layout.setStretch(2, 1)

    def init_functions_window(self):
        self.functions_layout = QVBoxLayout()
        self.caption_output_text = QTextEdit(self)
        self.caption_output_text.setReadOnly(True)
        self.functions_layout.addWidget(self.caption_output_text)

    def results_pkl_selected(self, index):
        item = self.results_pkl_select_cbox.itemText(index)
        self.results_pkl = item
        self.update_data_infos()

    def extrac_talk2event_sample(self):
        self.update_infos()
        self.data_dict = self.data_infos[self.sample_index]

    def on_frame_selected(self, item):
        self.sample_index = self.frame_list.row(item)
        self.extrac_talk2event_sample()
        self.show_sample()

    def update_data_infos(self):
        with open(os.path.join('/data/yyang/workspace/magiclidar/exps/record_results', self.results_pkl), 'rb') as f:
            self.data_infos = pickle.load(f)
        self.frame_list.clear()
        self.frame_list.addItems([f'sample_{i}' for i in range(len(self.data_infos))])

    def check_index_overflow(self) -> None:

        if self.sample_index == -1:
            self.sample_index = len(self.data_infos) - 1

        if self.sample_index >= len(self.data_infos):
            self.sample_index = 0

        self.frame_list.setCurrentRow(self.sample_index)

    def goto_sample_index(self):
        try:
            index = int(self.goto_sample_index_box.text())
            self.sample_index = index
            self.check_index_overflow()
            self.extrac_talk2event_sample()
            self.show_sample()
        except:
            raise NotImplementedError('Check the index.')

    def increment_index(self):
        self.sample_index += 1
        self.check_index_overflow()
        self.extrac_talk2event_sample()
        self.show_sample()

    def decrement_index(self):
        self.sample_index -= 1
        self.check_index_overflow()
        self.extrac_talk2event_sample()
        self.show_sample()

    def frame_increment_index(self):
        self.increment_index()

    def frame_decrement_index(self):
        self.decrement_index()

    def load_image(self):
        img_path = self.data_dict['image_path']
        bbox_2d = np.array(self.data_dict['gt_box'])
        class_2d = self.data_dict['gt_class']
        pred_box_2d = self.data_dict['pred_box']
        all_boxes = np.stack([bbox_2d, pred_box_2d], axis=0)
        img_w_box_2d = image_utils.load_img(img_path, all_boxes, class_2d)
        self.current_pixmap = gl_engine.ndarray_to_pixmap(img_w_box_2d)

        # self.update_image()

    def load_event(self):
        event_path = self.data_dict['event_path']
        bbox_2d = np.array(self.data_dict['gt_box'])
        class_2d = self.data_dict['gt_class']
        pred_box_2d = self.data_dict['pred_box']
        all_boxes = np.stack([bbox_2d, pred_box_2d], axis=0)
        event_w_box_2d = image_utils.load_event(event_path, all_boxes, class_2d)
        self.current_event_pixmap = gl_engine.ndarray_to_pixmap(event_w_box_2d)

        # self.update_image()

    def load_image_event(self):
        self.load_image()
        self.load_event()
        self.update_image()

    def update_infos(self):
        self.sample_index_info.setText(f"{self.sample_index}/{len(self.data_infos)}")

    def resizeEvent(self, event):
        if self.current_pixmap:
            self.update_image()

    def update_image(self):
        if self.current_pixmap:
            pixmap = self.current_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        if self.current_event_pixmap:
            pixmap = self.current_event_pixmap.scaled(self.event_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.event_label.setPixmap(pixmap)

    def save_sample(self):
        save_path = 'temp_test/inference_results'
        save_id = len(os.listdir(save_path)) // 2
        # image
        self.current_pixmap.save(os.path.join(save_path, f"{str(save_id).zfill(5)}_rgb.jpg"), "JPG", quality=100)
        # event
        self.current_event_pixmap.save(os.path.join(save_path, f"{str(save_id).zfill(5)}_event.jpg"), "JPG", quality=100)

    def show_caption(self):
        caption = self.data_dict['caption']
        self.caption_output_text.setText(caption)

    def show_sample(self):
        # image
        self.load_image()
        # event
        self.load_event()
        # caption
        self.show_caption()