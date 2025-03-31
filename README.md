PyLAD Viewer
============

Output visualization software for the [PyLAD (Python for Large Area Detectors) library](https://github.com/psavery/PyLAD).

![pylad-viewer-example](https://github.com/user-attachments/assets/fec5d433-d438-4b47-8c4b-8000307f6328)

This software was used at SLAC National Accelerator Laboratory at MEC during
experiments in early 2025.

## Description

PyLAD-Viewer is primarily intended to be used during live experimental
beamtimes in order to assist with the quick decision-making needed when
performing shots. It is opened and updated automatically by the PyLAD
client during beamtime as soon as new data becomes available
(see [here](https://github.com/psavery/PyLAD/blob/main/scripts/MEC/REAMDE.md)),
which allows the users to quickly discern what steps to take next.

PyLAD-Viewer displays output image files from the Varex detectors using a
few different methods: raw, flat and polar. All of the views have configuration
settings which may be edited in the [configuration file](#configuration).

All views may be zoomed/panned and have editable histograms and colormaps.
Mouse hover information (including the position and intensity) is displayed
in the bottom left corner of the application.

### Raw View

The raw view is contained in the top-left and top-middle tiles.

The raw view displays the raw (completely unmodified) detector images.
These are present for several use cases, including
determining whether a pre-shot contains saturated pixels
(see [Saturation Check](#saturation-check)).

The colormaps are by default linked so that relative intensities between the
two images can be compared. However, they may be unlinked by right-clicking
the histogram and clicking "unlink histograms and color maps".

### Flat View

The flat view is displayed in the top-right tile.

It is a view from the position of the sample looking toward the detectors
(imagine you are the sample looking at the detectors). It can be helpful
for understanding why the diffraction pattern may appear a certain way.

### Polar View

The polar view is in the bottom tile.

The polar view is warped so that two theta is along the x-axis, and eta
(or phi) is along the y axis. If the instrument is [calibrated](#configuration)
correctly, Debye-Scherrer rings should appear vertical.

At the bottom of the polar view is an azimuthal average (lineout) along two
theta. This shows intensities found at different values of two theta, and
is often essential for quickly discerning the state of the material at the
time the images were taken.

## Installing

This software must be installed from source. The repository must first by either
downloaded or cloned as follows:

```bash
git clone https://github.com/psavery/pylad-viewer
```

The code will be present in the `pylad-viewer` directory.

### Conda

We recommend creating and activating a conda environment to use with this
software.

Afterward, all dependencies must be installed, like so:

```bash
conda install -c hexrd -c conda-forge hexrd numpy pillow pyside6 pyqtgraph
```

Next, `pylad-viewer` must be installed as follows:

```bash
pip install --no-build-isolation --no-deps -U -e pylad-viewer
```

This installs `pylad-viewer` in an editable development environment, which
is helpful for doing things such as modifying the config file and instrument
file.

After installing, an example may be displayed by running `pylad-viewer`. Two
Varex detector files may also be opened by specifying their image file paths
as follows:

`pylad-viewer ./path/to/varex1.tiff ./path/to/varex2.tiff`

## Configuration

Two files may be edited to modify the settings of the application:
the instrument file and the configuration file.

The instrument file is a [HEXRD Instrument Configuration File](https://hexrdgui.readthedocs.io/en/latest/configuration/instrument/),
which specifies the position and orientation of the detectors,
beam vector information, etc. This instrument file is used for
generating the flat view and polar view. Calibrating the instrument
for your specific instrumental setup is necessary for the polar view
lineout to appear correct.

The instrument file is located in the source directory at
[pylad\_viewer/resources/instruments/MEC\_Varex.yml](https://github.com/psavery/PyLAD-Viewer/blob/main/pylad_viewer/resources/instruments/MEC_Varex.yml),
and it may be edited/replaced to update it.

The config file is located in the source directory at
[pylad\_viewer/resources/config.yml](https://github.com/psavery/PyLAD-Viewer/blob/main/pylad_viewer/resources/config.yml),
and it's contents may be edited to update it.

## Saturation Check

A checkbox in the top-left corner of the application may be used to
perform a saturation check. This is helpful to use during
pre-shot to determine if there are any spots that are too intense
(e.g., LiF spots) which may damage the detector at full x-ray
strength.

When checked, a message will appear indicating the
number of saturated pixels detected.

![saturation_warning](https://github.com/user-attachments/assets/f3a4d612-6c0d-4ab5-bf17-bc6363682438)

The colormap settings are also
automatically adjusted so that saturated pixels are black and all
other pixels are white. Red circles are drawn around saturated
pixels so they can be more easily located.

![saturation_check1](https://github.com/user-attachments/assets/38d862aa-b158-4503-95c4-1f7c10236b7c)

![saturation_check2](https://github.com/user-attachments/assets/1fc55995-d4fb-4b1d-a676-3ed5f200f37e)

The saturation level may be modified in the config settings.
