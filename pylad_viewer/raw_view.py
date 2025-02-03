from contextlib import contextmanager
from functools import partial

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QWidget,
)

import numpy as np
import pyqtgraph as pg


class RawImagesWidget(QWidget):

    mouse_move_message = Signal(str)

    def __init__(self, array_list: list[np.ndarray], parent=None):
        super().__init__(parent)

        self.setup_widgets(len(array_list))
        self.set_data(array_list)

        self.link_levels_and_cmaps = True
        self._updating_histograms = False

        self.setup_connections()

        # Reverse one of the cmaps. Since the others are linked,
        # they will be reversed too.
        self.reverse_cmap(self.histogram_widgets[0])

    def setup_connections(self):
        for im in self.image_view_list:
            f = partial(self._on_mouse_move, image_view=im)
            im.scene.sigMouseMoved.connect(f)

        for hist in self.histogram_widgets:
            hist.sigLookupTableChanged.connect(self._on_lookup_table_changed)
            hist.sigLevelsChanged.connect(self._on_levels_changed)

    def setup_widgets(self, num_arrays):
        # container widget with a layout to add QWidgets to
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.image_view_list = []
        for _ in range(num_arrays):
            im = pg.ImageView()
            layout.addWidget(im)
            self.image_view_list.append(im)

        self.add_additional_context_menu_actions()

        # Uncomment these to also link image views and histogram ranges
        # self.link_image_views()
        # self.link_histogram_ranges()

        self.reflection_artists = []
        for image_view in self.image_view_list:
            artist = pg.ScatterPlotItem(
                pxMode=True,
                symbol='o',
                pen=pg.mkPen(0, 1, width=1),
                brush=None,
                size=5,
            )
            self.reflection_artists.append(artist)
            image_view.addItem(artist)

    def set_data(self, array_list: list[np.ndarray]):
        self.array_list = array_list

        for im, array in zip(self.image_view_list, array_list):
            im.setImage(array)

        self.auto_level_colors()
        self.auto_level_histogram_range()

    def add_additional_context_menu_actions(self):
        for hist in self.histogram_widgets:
            self.add_additional_cmap_menu_actions(hist)
            self.add_additional_histogram_menu_actions(hist)

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

    def add_additional_histogram_menu_actions(self, w):
        """Add a 'auto level' action to the pyqtgraph histogram menu

        This assumes pyqtgraph won't change its internal attribute structure.
        If it does change, then this function just won't work...
        """
        try:
            vb = w.item.vb
            menu = vb.menu
        except AttributeError:
            # pyqtgraph must have changed its attribute structure
            return

        if not menu:
            return

        def auto_level():
            self.auto_level_colors()
            self.auto_level_histogram_range()

        menu.addSeparator()
        action = menu.addAction('auto level')
        action.triggered.connect(auto_level)

        action = menu.addAction('unlink histograms and color maps')

        def toggle_link():
            if self.link_levels_and_cmaps:
                self.link_levels_and_cmaps = False
                action.setText('link histograms and color maps')
            else:
                self.link_levels_and_cmaps = True
                action.setText('unlink histograms and color maps')

        action.triggered.connect(toggle_link)

    @property
    def histogram_widgets(self):
        return [im.getHistogramWidget() for im in self.image_view_list]

    def set_levels_for_saturation_check(self):
        lower = 38000
        upper = 38500
        for im in self.image_view_list:
            im.setLevels(lower, upper)
            im.setHistogramRange(lower - 1e3, upper + 1e3)

        data = np.array(self.array_list)
        num_saturated = np.count_nonzero(data > lower)
        if num_saturated > 0:
            for array, artist in zip(self.array_list, self.reflection_artists):
                coords = np.argwhere(array > lower)
                if coords.size == 0:
                    continue

                coords = coords.astype(float) + 0.5
                coords = np.atleast_2d(coords)[:, [1, 0]]
                artist.setData(*coords.T)

            QMessageBox.critical(
                None,
                'Saturation Warning',
                f'Data contains {num_saturated} pixels '
                f'above {lower} in value'
            )

    def clear_artists(self):
        for artist in self.reflection_artists:
            artist.clear()

    def auto_level_colors(self):
        # These levels appear to work well for the data we have
        data = np.array(self.array_list)
        lower = np.nanpercentile(data, 1.0)
        upper = np.nanpercentile(data, 99.75)

        for im in self.image_view_list:
            im.setLevels(lower, upper)

    def auto_level_histogram_range(self):
        # Make the histogram range a little bigger than the auto level colors
        data = np.array(self.array_list)
        lower = np.nanpercentile(data, 0.5)
        upper = np.nanpercentile(data, 99.8)

        for im in self.image_view_list:
            im.setHistogramRange(lower, upper)

    def link_image_views(self):
        im1_view = self.image_view_list[0].getView()
        for im in self.image_view_list[1:]:
            view = im.getView()
            view.setXLink(im1_view)
            view.setYLink(im1_view)

    def unlink_image_views(self):
        for im in self.image_view_list:
            view = im.getView()
            view.setXLink(None)
            view.setYLink(None)

    def link_histogram_ranges(self):
        histogram_widgets = self.histogram_widgets
        hist1 = histogram_widgets[0]
        for hist in histogram_widgets[1:]:
            hist.item.vb.setYLink(hist1.item.vb)

    def unlink_histogram_ranges(self):
        for hist in self.histogram_widgets:
            hist.item.vb.setYLink(None)

    def _on_lookup_table_changed(self, changed_hist_item):
        if self._updating_histograms or not self.link_levels_and_cmaps:
            # To avoid infinite recursion, don't trigger this function
            # again if we are already updating histograms.
            return

        cmap = changed_hist_item.gradient.colorMap()
        histograms = [x for x in self.histogram_widgets
                      if x.item is not changed_hist_item]
        with self.updating_histograms_on:
            for hist in histograms:
                hist.item.gradient.setColorMap(cmap)

    def _on_levels_changed(self, changed_hist_item):
        if self._updating_histograms or not self.link_levels_and_cmaps:
            # To avoid infinite recursion, don't trigger this function
            # again if we are already updating histograms.
            return

        levels = changed_hist_item.getLevels()
        histograms = [x for x in self.histogram_widgets
                      if x.item is not changed_hist_item]
        with self.updating_histograms_on:
            for hist in histograms:
                hist.item.setLevels(*levels)

    def _on_mouse_move(self, pos, image_view):
        data = image_view.getImageItem().image

        # First, map the scene coordinates to the view
        pos = image_view.view.mapSceneToView(pos)

        # We get the correct pixel coordinates by flooring these
        j, i = np.floor(pos.toTuple()).astype(int)

        data_shape = data.shape
        if not 0 <= i < data_shape[0] or not 0 <= j < data_shape[1]:
            # The mouse is out of bounds
            return

        # For display, x and y are the same as j and i, respectively
        x, y = j, i

        intensity = data[i, j]

        # Unfortunately, if we do f'{x=}', it includes the numpy
        # dtype, which we don't want.
        message = f'x={x}, y={y}, intensity={intensity}'
        self.mouse_move_message.emit(message)

    @property
    @contextmanager
    def updating_histograms_on(self):
        prev = self._updating_histograms
        self._updating_histograms = True
        try:
            yield
        finally:
            self._updating_histograms = prev


if __name__ == '__main__':

    from pathlib import Path
    import sys

    from PySide6.QtWidgets import QMainWindow

    from PIL import Image

    import pylad_viewer

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

    pg.setConfigOptions(
        **{
            # Use row-major for the imageAxisOrder in pyqtgraph
            'imageAxisOrder': 'row-major',
            # Use numba acceleration where we can
            'useNumba': True,
        }
    )

    app = pg.mkQApp()

    varex1 = np.array(Image.open(image_file_paths[0]))
    varex2 = np.array(Image.open(image_file_paths[1]))

    win = QMainWindow()

    def show_message(msg):
        win.statusBar().showMessage(msg)

    raw_images_widget = RawImagesWidget([varex1, varex2], win)
    raw_images_widget.mouse_move_message.connect(show_message)

    win.setCentralWidget(raw_images_widget)

    win.show()
    app.exec()
