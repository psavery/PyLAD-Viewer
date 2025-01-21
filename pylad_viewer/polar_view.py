from functools import partial

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

import numpy as np
import pyqtgraph as pg

from hexrd.projections.polar import PolarView


class PolarViewWidget(QWidget):

    mouse_move_message = Signal(str)

    def __init__(
        self,
        pv: PolarView,
        image_dict: dict[np.ndarray],
        parent=None,
    ):
        super().__init__(parent)

        # These are perfect for our Varex setup. Maybe we should expose
        # them as an option sometime for other types of detectors.
        default_y_range = [-95, 95]

        self.pv = pv
        self.image_dict = image_dict

        # First, create the polar view image and set it
        polar_img = pv.warp_image(
            image_dict,
            pad_with_nans=True,
            do_interpolation=True,
        )

        layout = QVBoxLayout()
        self.setLayout(layout)

        image_label_plot = pg.PlotItem()
        image_label_plot.setLabel('left', '<font> &eta; </font>', units='deg')

        im = pg.ImageView(view=image_label_plot)
        im.setImage(polar_img.filled(np.nan))
        im.view.setAspectLocked(False)
        im.view.vb.setBackgroundColor('white')
        self.image_view = im

        extent = np.degrees(pv.extent)
        im.imageItem.setRect(
            extent[0],
            extent[3],
            extent[1] - extent[0],
            extent[2] - extent[3],
        )
        layout.addWidget(im, stretch=3)

        # Next, create the lineout and set it
        lineout = polar_img.sum(axis=0) / np.sum(~polar_img.mask, axis=0)

        plt = pg.plot()
        plt.showGrid(x=True, y=True)
        self.lineout_plot = plt

        plt.setLabel('left', 'Azimuthal Average')
        plt.setLabel('bottom', '<font> 2&theta; </font>', units='deg')

        tth_values = pv.angular_grid[1][0]
        plt.plot(np.degrees(tth_values), lineout)
        layout.addWidget(plt, stretch=1)

        im.view.setRange(
            xRange=(*extent[:2],),
            yRange=default_y_range,
            padding=0.01,
        )

        plt.setXLink(im.view)

        self.add_additional_context_menu_actions()
        self.reverse_cmap(self.histogram_widget)

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

    tth_range = [4.0, 95.0]
    eta_min = -180.0
    eta_max = 180.0
    # pixel_size = (0.01, 0.1)
    pixel_size = (0.1, 0.1)

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

    polar_view_widget = PolarViewWidget(pv, img_dict, win)
    polar_view_widget.mouse_move_message.connect(show_message)

    win.setCentralWidget(polar_view_widget)

    win.show()
    app.exec()
