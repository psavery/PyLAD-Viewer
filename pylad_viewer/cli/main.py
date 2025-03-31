import importlib.resources
from pathlib import Path
import sys
import threading

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
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
import pylad_viewer.resources.instruments
from pylad_viewer.instrument_projection import InstrumentProjection
from pylad_viewer.flat_view import FlatViewWidget
from pylad_viewer.raw_view import RawImagesWidget
from pylad_viewer.polar_view import PolarViewWidget


def main():
    if len(sys.argv) == 1:
        # Use the default images
        repo_dir = Path(pylad_viewer.__file__).parent.parent
        ceria_example_path = repo_dir / 'examples/ceria'
        images_path = ceria_example_path / 'Run1'
        image_file_paths = [
            images_path / 'varex1.tif',
            images_path / 'varex2.tif',
        ]
        # Use the example instrument
        instr_filename = 'Example_MEC_Varex.yml'
        config_filename = 'example_config.yml'
    else:
        image_file_paths = sys.argv[1:3]
        # Use the latest instrument file
        instr_filename = 'MEC_Varex.yml'
        config_filename = 'config.yml'

    resources_path = importlib.resources.files(
        pylad_viewer.resources)

    instruments_path = importlib.resources.files(
        pylad_viewer.resources.instruments)

    repo_dir = Path(pylad_viewer.__file__).parent.parent
    ceria_example_path = repo_dir / 'examples/ceria'

    # Load the instrument
    with open(instruments_path / instr_filename, 'r') as rf:
        iconf = yaml.safe_load(rf)

    instr = HEDMInstrument(iconf)

    with open(instruments_path / 'varex_flat_projection.yml', 'r') as rf:
        iconf = yaml.safe_load(rf)

    flat_instr = HEDMInstrument(iconf)

    # Open up the config
    with open(resources_path / config_filename, 'r') as rf:
        conf = yaml.safe_load(rf)

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

    def show_message(msg):
        win.statusBar().showMessage(msg)

    top_bar_layout = QHBoxLayout()

    v_layout = QVBoxLayout()
    top_layout = QHBoxLayout()
    bottom_layout = QHBoxLayout()

    check_saturation_cb = QCheckBox('Saturation Check')
    check_saturation_cb.setToolTip(
        'Perform a saturation check. This is helpful to use during '
        'pre-shot to determine if there are any spots that are too intense '
        '(e.g., LiF spots) which may damage the detector at full x-ray '
        'strength.\n\nWhen checked, a message will appear indicating the '
        'number of saturated pixels detected. The colormap settings are also '
        'automatically adjusted so that saturated pixels are black and all '
        'other pixels are white. Red circles are drawn around saturated '
        'pixels so they can be more easily located.\n\nThe saturation level '
        'may be modified in the config settings.'
    )
    run_num_label = QLabel()
    run_num_label.setStyleSheet(
        'font-size: 18pt; font-weight: bold; color: red'
    )

    def set_title(title):
        win.setWindowTitle(title)
        run_num_label.setText(title)

    set_title(Path(image_file_paths[0]).parent.name)

    top_bar_layout.addWidget(check_saturation_cb, 1, Qt.AlignLeft)
    top_bar_layout.addWidget(run_num_label, 1, Qt.AlignHCenter)
    top_bar_layout.addWidget(QLabel(), 1, Qt.AlignRight)

    v_layout.addLayout(top_bar_layout)
    v_layout.addLayout(top_layout, stretch=2)
    v_layout.addLayout(bottom_layout, stretch=3)

    central_widget = QWidget()
    central_widget.setLayout(v_layout)

    win.setCentralWidget(central_widget)

    # Raw images view
    raw_images_widget = RawImagesWidget(list(img_dict.values()), win)
    raw_images_widget.mouse_move_message.connect(show_message)
    top_layout.addWidget(raw_images_widget, stretch=3)
    raw_images_widget.saturation_level = conf['raw_view']['saturation_level']

    def check_saturation_toggled(b: bool):
        if b:
            raw_images_widget.set_levels_for_saturation_check()
        else:
            raw_images_widget.clear_artists()
            raw_images_widget.auto_level_colors()
            raw_images_widget.auto_level_histogram_range()

    check_saturation_cb.toggled.connect(check_saturation_toggled)

    # Flat view
    tth_range = conf['flat_view']['tth_range']
    pixel_size = conf['flat_view']['pixel_sizes']
    # I don't think we need to ever change these eta ranges. If we
    # do, we can add it as an option in the config.
    eta_min = 0.0
    eta_max = 360.0

    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size,
                   cache_coordinate_map=True)

    ip = InstrumentProjection(instr, flat_instr, pv)
    flat_view_widget = FlatViewWidget(ip, img_dict, win)
    flat_view_widget.mouse_move_message.connect(show_message)
    top_layout.addWidget(flat_view_widget, stretch=2)

    # Polar view
    tth_range = conf['polar_view']['tth_range']
    eta_min, eta_max = conf['polar_view']['eta_range']
    pixel_size = conf['polar_view']['pixel_sizes']

    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size,
                   cache_coordinate_map=True)

    polar_view_widget = PolarViewWidget(pv, img_dict, win)
    polar_view_widget.mouse_move_message.connect(show_message)
    bottom_layout.addWidget(polar_view_widget)

    new_images_stack = []
    new_images_lock = threading.Lock()

    def set_data(filepaths):
        # This ought to be ran on the GUI thread.
        print('Setting data with filepaths:', filepaths, flush=True)
        set_title('Loading new data...')
        img_dict = create_image_dict(filepaths)

        raw_images_widget.set_data(list(img_dict.values()))
        flat_view_widget.set_data(img_dict)
        polar_view_widget.set_data(img_dict)

        check_saturation_cb.setChecked(False)

        set_title(Path(filepaths[0]).parent.name)

    def check_for_new_images_loop():
        # This runs in the GUI thread, and just checks for new images
        # in a mutex-controlled list.
        new_images = None
        with new_images_lock:
            if new_images_stack:
                new_images = new_images_stack.pop(0)

        if new_images:
            set_data(new_images)

        QTimer.singleShot(0.5, check_for_new_images_loop)

    def read_stdin_loop():
        # This runs in a separate thread.
        # Read from stdin every 0.5 seconds to see if new datasets were
        # provided.
        # If new images are found, add them to a mutex-controlled list
        for message in sys.stdin:
            message = message.rstrip()
            if message:
                print('Message received from stdin:', message, flush=True)
                if ', ' in message:
                    filepaths = message.split(', ')

                    if any(not Path(x).exists() for x in filepaths):
                        print('Some file paths did not exist. Ignoring...')
                        continue

                    # Run this in the GUI thread
                    # There's another loop listening to this images stack
                    with new_images_lock:
                        new_images_stack.append(filepaths)

    # FIXME: this is sort of complicated, but I couldn't get other methods
    # to work on Windows. We have a separate thread that constantly reads
    # from stdin, and it posts new images to a mutex-controlled list, which
    # is checked regularly by the GUI thread.

    # Start the stdin reader thread
    thread = threading.Thread(target=read_stdin_loop)
    thread.start()
    check_for_new_images_loop()

    win.showMaximized()
    app.exec()
