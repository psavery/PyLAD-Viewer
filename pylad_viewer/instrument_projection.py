import numpy as np
from scipy.interpolate import RegularGridInterpolator

from hexrd.instrument import HEDMInstrument
from hexrd.projections.polar import PolarView
from hexrd.rotations import mapAngle


class InstrumentProjection:

    def __init__(
        self,
        instr: HEDMInstrument,
        projection_instr: HEDMInstrument,
        pv: PolarView,
    ):
        self.instr = instr
        self.projection_instr = projection_instr
        self.pv = pv

    def make_projected_image(self, img_dict):
        '''main routine to warp the cake image
        back to an equivalent detector image
        '''
        pvarray = self.pv.warp_image(img_dict,
                                     pad_with_nans=True,
                                     do_interpolation=True)
        pvarray2 = pvarray.data
        pvarray2[pvarray.mask] = np.nan
        self.pvarray = pvarray2

        self.initialize_interpolation_object()
        self.project_intensities_to_raw()

    def initialize_interpolation_object(self):
        kwargs = {
            'points': (self.eta_grid, self.tth_grid),
            'values': self.pvarray,
            'method': 'linear',
            'bounds_error': False,
            'fill_value': np.nan,
        }

        self.interp_obj = RegularGridInterpolator(**kwargs)

    def project_intensities_to_raw(self):
        self.projected_image = dict.fromkeys(self.projection_instr.detectors)
        for det_name, det in self.projection_instr.detectors.items():
            img = self.project_intensity_detector(det)
            self.projected_image[det_name] = img

    def project_intensity_detector(self,
                                   det):
        tth, eta = np.degrees(det.pixel_angles())
        eta = mapAngle(eta, (0, 360.0), units='degrees')
        xi = (eta, tth)
        return self.interp_obj(xi)

    @property
    def eta_grid(self):
        return np.degrees(self.pv.angular_grid[0][:, 0])

    @property
    def tth_grid(self):
        return np.degrees(self.pv.angular_grid[1][0, :])
