import importlib.resources
from pathlib import Path
import select
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

import numpy as np
from PIL import Image
import pyqtgraph as pg
import yaml

from hexrd.instrument import HEDMInstrument
from hexrd.projections.polar import PolarView

import pylad_viewer
import pylad_viewer.resources
from pylad_viewer.instrument_projection import InstrumentProjection
from pylad_viewer.flat_view import FlatViewWidget
from pylad_viewer.raw_view import RawImagesWidget
from pylad_viewer.polar_view import PolarViewWidget


def main():
    if len(sys.argv) == 1:
        # Use the default images
        repo_dir = Path(pylad_viewer.__file__).parent.parent
        ceria_example_path = repo_dir / 'examples/ceria'
        images_path = ceria_example_path / 'images'
        image_file_paths = [
            images_path / 'varex1.tif',
            images_path / 'varex2.tif',
        ]
    else:
        image_file_paths = sys.argv[1:3]

    resources_path = importlib.resources.files(pylad_viewer.resources)

    repo_dir = Path(pylad_viewer.__file__).parent.parent
    ceria_example_path = repo_dir / 'examples/ceria'

    # Load the instrument
    with open(resources_path / 'MEC_Varex.yml', 'r') as rf:
        conf = yaml.safe_load(rf)

    instr = HEDMInstrument(conf)

    with open(resources_path / 'varex_flat_projection.yml', 'r') as rf:
        conf = yaml.safe_load(rf)

    flat_instr = HEDMInstrument(conf)

    pg.setConfigOptions(
        **{
            # Use row-major for the imageAxisOrder in pyqtgraph
            'imageAxisOrder': 'row-major',
            # Use numba acceleration where we can
            'useNumba': True,
        }
    )

    app = pg.mkQApp()

    def create_image_dict(filepaths):
        return {
            'Varex1': np.array(Image.open(filepaths[0])),
            'Varex2': np.array(Image.open(filepaths[1])),
        }

    img_dict = create_image_dict(image_file_paths)

    win = QMainWindow()
    win.setWindowTitle(Path(image_file_paths[0]).parent.name)

    def show_message(msg):
        win.statusBar().showMessage(msg)

    v_layout = QVBoxLayout()
    top_layout = QHBoxLayout()
    bottom_layout = QHBoxLayout()

    check_saturation_cb = QCheckBox('Saturation Check')
    v_layout.addWidget(check_saturation_cb)

    v_layout.addLayout(top_layout, stretch=2)
    v_layout.addLayout(bottom_layout, stretch=3)

    central_widget = QWidget()
    central_widget.setLayout(v_layout)

    win.setCentralWidget(central_widget)

    # Raw images view
    raw_images_widget = RawImagesWidget(list(img_dict.values()), win)
    raw_images_widget.mouse_move_message.connect(show_message)
    top_layout.addWidget(raw_images_widget, stretch=3)

    def check_saturation_toggled(b: bool):
        if b:
            raw_images_widget.set_levels_for_saturation_check()
        else:
            raw_images_widget.clear_artists()
            raw_images_widget.auto_level_colors()
            raw_images_widget.auto_level_histogram_range()

    check_saturation_cb.toggled.connect(check_saturation_toggled)

    # Flat view
    tth_range = [6.0, 86.0]
    eta_min = 0.0
    eta_max = 360.0
    # pixel_size = (0.025, 0.25)
    pixel_size = (0.05, 0.05)

    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size,
                   cache_coordinate_map=True)

    ip = InstrumentProjection(instr, flat_instr, pv)
    flat_view_widget = FlatViewWidget(ip, img_dict, win)
    flat_view_widget.mouse_move_message.connect(show_message)
    top_layout.addWidget(flat_view_widget, stretch=2)

    # Polar view
    tth_range = [12.0, 86.0]
    eta_min = -90.0
    eta_max = 270.0
    # pixel_size = (0.01, 0.1)
    pixel_size = (0.025, 0.1)

    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size,
                   cache_coordinate_map=True)

    polar_view_widget = PolarViewWidget(pv, img_dict, win)
    polar_view_widget.mouse_move_message.connect(show_message)
    bottom_layout.addWidget(polar_view_widget)

    def set_data(filepaths):
        print('Setting data with filepaths:', filepaths, flush=True)
        win.setWindowTitle('Loading new data...')
        img_dict = create_image_dict(filepaths)

        raw_images_widget.set_data(list(img_dict.values()))
        flat_view_widget.set_data(img_dict)
        polar_view_widget.set_data(img_dict)

        win.setWindowTitle(Path(filepaths[0]).parent.name)

    def read_stdin_loop():
        # Read from stdin every 0.5 seconds to see if new datasets were
        # provided.
        timeout = 0
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if sys.stdin in rlist:
            message = sys.stdin.readline().rstrip()
            if message:
                print('Message received from stdin:', message, flush=True)
                if ', ' in message:
                    filepaths = message.split(', ')
                    set_data(filepaths)

        QTimer.singleShot(0.5, read_stdin_loop)

    # Start the stdin reader loop
    # This checks stdin every 0.5 seconds to see if there was new
    # data provided.
    read_stdin_loop()

    win.showMaximized()
    app.exec()
