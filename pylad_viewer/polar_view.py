import copy
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

        self.pv = pv

        self.setup_widgets()
        self.set_data(image_dict)

        self.setup_connections()

    def setup_connections(self):
        self.image_view.scene.sigMouseMoved.connect(self.on_image_mouse_move)
        self.lineout_plot.scene().sigMouseMoved.connect(
            self.on_lineout_plot_move)

    def setup_widgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Image view
        image_label_plot = pg.PlotItem()
        image_label_plot.setLabel('left', '<font> &eta; </font>', units='deg')

        im = pg.ImageView(view=image_label_plot)
        im.view.setAspectLocked(False)

        layout.addWidget(im, stretch=3)
        self.image_view = im

        # Lineout plot
        plt = pg.plot()
        plt.showGrid(x=True, y=True)
        self.lineout_plot = plt

        plt.setLabel('left', 'Azimuthal Average')
        plt.setLabel('bottom', '<font> 2&theta; </font>', units='deg')
        layout.addWidget(plt, stretch=1)

        plt.setXLink(im.view)

        self.add_additional_context_menu_actions()
        self.reverse_cmap(self.histogram_widget)

    def set_data(self, image_dict: dict[str, np.ndarray]):
        # These are perfect for our Varex setup. Maybe we should expose
        # them as an option sometime for other types of detectors.
        default_y_range = [-90, 270]

        # self.unscaled_image_dict = image_dict

        # We will perform log scaling on the images
        image_dict = copy.deepcopy(image_dict)

        self.image_dict = image_dict

        # for key, img in image_dict.items():
        #     img -= np.nanmin(img)
        #     img = np.log(img + 1)
        #     image_dict[key] = img

        # First, create the polar view image and set it
        pv = self.pv
        polar_img = pv.warp_image(
            image_dict,
            pad_with_nans=True,
            do_interpolation=True,
        )

        # unscaled_polar_img = pv.warp_image(
        #     self.unscaled_image_dict,
        #     pad_with_nans=True,
        #     do_interpolation=True,
        # )

        self.image_view.setImage(polar_img.filled(np.nan))

        extent = np.degrees(pv.extent)
        self.image_view.imageItem.setRect(
            extent[0],
            extent[3],
            extent[1] - extent[0],
            extent[2] - extent[3],
        )

        # Next, create the lineout and set it
        lineout = (
            polar_img.sum(axis=0) /
            np.sum(~polar_img.mask, axis=0)
        )

        # Any columns that are all nans should just be nan
        lineout = lineout.filled(np.nan)

        tth_values = pv.angular_grid[1][0]
        self.lineout_plot.clear()
        self.lineout_plot.plot(np.degrees(tth_values), lineout)

        self.image_view.view.setRange(
            xRange=(*extent[:2],),
            yRange=default_y_range,
            padding=0.01,
        )

        self.auto_level_colors()
        self.auto_level_histogram_range()

    def on_image_mouse_move(self, pos):
        data = self.image_view.getImageItem().image

        # First, map the scene coordinates to the view
        pos = self.image_view.view.vb.mapSceneToView(pos)

        # We get the correct pixel coordinates by flooring these
        tth, eta = pos.toTuple()

        j = int(np.round(
            (tth - np.degrees(self.pv.tth_min)) / self.pv.tth_pixel_size - 0.5
        ))
        i = int(np.round(
            (eta - np.degrees(self.pv.eta_min)) / self.pv.eta_pixel_size - 0.5
        ))

        data_shape = data.shape
        if not 0 <= i < data_shape[0] or not 0 <= j < data_shape[1]:
            # The mouse is out of bounds
            return

        intensity = data[i, j]

        # Unfortunately, if we do f'{x=}', it includes the numpy
        # dtype, which we don't want.
        message = f'tth={tth:.3f}, eta={eta:.3f}, intensity={intensity:.3f}'
        self.mouse_move_message.emit(message)

    def on_lineout_plot_move(self, pos):
        pos = self.lineout_plot.plotItem.vb.mapSceneToView(pos)
        x, y = pos.toTuple()
        message = f'x={x:.3f}, y={y:.3f}'
        self.mouse_move_message.emit(message)

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
        data = [x.flatten() for x in self.array_list]
        lower = np.nanpercentile(data, 1.0)
        upper = np.nanpercentile(data, 99.9)

        self.image_view.setLevels(lower, upper)

    def auto_level_histogram_range(self):
        # Make the histogram range a little bigger than the auto level colors
        data = [x.flatten() for x in self.array_list]
        lower = np.nanpercentile(data, 0.5)
        upper = np.nanpercentile(data, 99.99)

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
