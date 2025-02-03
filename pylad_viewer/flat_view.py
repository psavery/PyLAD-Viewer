from functools import partial

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

import numpy as np
import pyqtgraph as pg

from hexrd.projections.polar import PolarView

from pylad_viewer.instrument_projection import InstrumentProjection


class FlatViewWidget(QWidget):

    mouse_move_message = Signal(str)

    def __init__(
        self,
        ip: InstrumentProjection,
        image_dict: dict[np.ndarray],
        parent=None,
    ):
        super().__init__(parent)

        self.ip = ip
        self.image_view = None

        self.setup_image_view()
        self.set_data(image_dict)

    def setup_image_view(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view)

        self.add_additional_context_menu_actions()
        self.reverse_cmap(self.histogram_widget)

    def set_data(self, image_dict: dict[str, np.ndarray]):
        self.image_dict = image_dict

        # First, create the projected image and set it
        self.ip.make_projected_image(image_dict)

        self.image_view.setImage(self.projected_image)

        self.auto_level_colors()
        self.auto_level_histogram_range()

    @property
    def projected_image(self) -> np.ndarray:
        return next(iter(self.ip.projected_image.values()))

    @property
    def histogram_widget(self):
        return self.image_view.getHistogramWidget()

    def add_additional_context_menu_actions(self):
        self.add_additional_cmap_menu_actions(self.histogram_widget)

    def add_additional_cmap_menu_actions(self, w):
        """Add a 'reverse' action to the pyqtgraph colormap menu

        This assumes pyqtgraph won't change its internal attribute structure.
        If it does change, then this function just won't work...
        """
        try:
            gradient = w.item.gradient
            menu = gradient.menu
        except AttributeError:
            # pyqtgraph must have changed its attribute structure
            return

        if not menu:
            return

        reverse = partial(self.reverse_cmap, hist=w)

        menu.addSeparator()
        action = menu.addAction('reverse')
        action.triggered.connect(reverse)

    def reverse_cmap(self, hist):
        gradient = hist.item.gradient
        cmap = gradient.colorMap()
        cmap.reverse()
        gradient.setColorMap(cmap)

    @property
    def array_list(self) -> list[np.ndarray]:
        return list(self.image_dict.values())

    def auto_level_colors(self):
        # These levels appear to work well for the data we have
        data = self.projected_image
        lower = np.nanpercentile(data, 1.0)
        upper = np.nanpercentile(data, 99.75)

        self.image_view.setLevels(lower, upper)

    def auto_level_histogram_range(self):
        # Make the histogram range a little bigger than the auto level colors
        data = self.projected_image
        lower = np.nanpercentile(data, 0.5)
        upper = np.nanpercentile(data, 99.8)

        self.image_view.setHistogramRange(lower, upper)


if __name__ == '__main__':

    import importlib.resources
    from pathlib import Path

    from PySide6.QtWidgets import QMainWindow

    from PIL import Image
    import yaml

    from hexrd.instrument import HEDMInstrument

    import pylad_viewer.resources

    resources_path = importlib.resources.files(pylad_viewer.resources)

    repo_dir = Path(pylad_viewer.__file__).parent.parent
    ceria_example_path = repo_dir / 'examples/ceria'
    images_path = ceria_example_path / 'images'

    # Load the instrument
    with open(resources_path / 'MEC_Varex.yml', 'r') as rf:
        conf = yaml.safe_load(rf)

    instr = HEDMInstrument(conf)

    with open(resources_path / 'varex_flat_projection.yml', 'r') as rf:
        conf = yaml.safe_load(rf)

    flat_instr = HEDMInstrument(conf)

    tth_range = [6.0, 72.0]
    eta_min = 0.0
    eta_max = 360.0
    # pixel_size = (0.025, 0.25)
    pixel_size = (0.05, 0.05)

    pv = PolarView(tth_range, instr, eta_min, eta_max, pixel_size,
                   cache_coordinate_map=True)

    pg.setConfigOptions(
        **{
            # Use row-major for the imageAxisOrder in pyqtgraph
            'imageAxisOrder': 'row-major',
            # Use numba acceleration where we can
            'useNumba': True,
        }
    )

    app = pg.mkQApp()

    img_dict = {
        'Varex1': np.array(Image.open(images_path / 'varex1.tif')),
        'Varex2': np.array(Image.open(images_path / 'varex2.tif')),
    }

    win = QMainWindow()

    def show_message(msg):
        win.statusBar().showMessage(msg)

    ip = InstrumentProjection(instr, flat_instr, pv)
    flat_view_widget = FlatViewWidget(ip, img_dict, win)
    flat_view_widget.mouse_move_message.connect(show_message)

    win.setCentralWidget(flat_view_widget)

    win.show()
    app.exec()
