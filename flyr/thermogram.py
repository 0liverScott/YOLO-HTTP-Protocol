""" The main public API to the data in a FLIR thermogram and the result of
     `flyr.unpack()`.
"""
import os
import warnings
import numpy as np

from PIL import Image
from nptyping import Array
from math import sqrt, exp, fsum
from typing import List, Optional, Dict, Tuple, Union

import flyr.palettes as palettes
import flyr.normalization as norm
import flyr.palette_info as pal
import flyr.picture_in_picture as pip


class FlyrThermogram:
    """ A FlyrThermogram is a class providing read-only access to the data in a typical
        FLIR thermogram.

        Specifically interesting are:

        * `kelvin` (property): Getting the temperature in degrees kelvin
        * `celsius` (property): Getting the temperature in degrees celsius
        * `metadata` (property): Getting the FLIR camera metadata, some of which
            influences how `kelvin`/`celsius` are calculated.
        * `adjust_metadata()` (method): Updates the above-mentioned metadata and can thus
            be used to change how the temperature is calculated.
        * `render()` (method): Returns the temperature as an RGB array. Use the
            `render_pil` variant to get one
    """

    # Required members variables required to be set
    __thermal: Array[np.int64, ..., ...]
    __metadata: Dict[str, Union[float, int]]
    __metadata_adjustments: Dict[str, Union[float, int]]
    __optical: Optional[Array[int, ..., ..., 3]]
    palette: Optional[pal.Palette]
    pip_info: Optional[pip.PictureInPicture]
    path: Optional[str]

    def __init__(
        self,
        thermal: Array[np.int64, ..., ...],
        metadata: Dict[str, Union[float, int]],
        optical: Optional[Array[np.uint8, ..., ..., 3]] = None,
        palette: Optional[pal.Palette] = None,
        picture_in_picture: Optional[pip.PictureInPicture] = None,
        path: Optional[str] = None,
        metadata_adjustments: Dict[str, Union[float, int]] = {},
    ):
        """ Initialize a new instance of this class. The raw thermal data and
            the accompanying metadata to correctly interpret this data are
            required.

            Parameters
            ----------
            thermal: Array[np.int64, ..., ...]
                A 2D numpy array of 64 bit integers. This is the raw thermal
                data as it is stored in the FLIR file. Order is [H, W].
            metadata: Dict[str, Union[float, int]]
                A dictionary with physical parameters to interpret the raw
                thermal data, as it was included in the original FLIR file.
            optical: Array[np.uint8, ..., ..., 3]
                A 3D numpy array of 8 bit integers, in the order of [H, W, C].
                This should be a 'normal' photo (RGB) of the same scene as
                thermogram.
            palette: Optional[pinfo.Palette]
                The palette embedded in the file, which can be used to render
                with by default. Is `None` by default in which case a grayscale
                palette is used.
            picture_in_picture: Optional[pip.PictureInPicture]
                The picture-in-picture settings useful for stacking optical and
                thermal data.
            path: Optional[str]
                The path to the original file. Is only used to determine this
                this object's identifier and can safely be left `None`
                (default).
            metadata_adjustments: Dict[str, Union[float, int]]
                A dictionary with adjustments to the above dictionary. This
                separation allows both using the original settings and
                calculating temperatures with different physical parameters.

            Returns
            -------
            FlyrThermogram
        """
        self.__thermal = thermal  # Raw thermal data
        self.__optical = optical  # Optical (RGB) photo in the thermogram
        self.__metadata = metadata.copy()
        self.__metadata_adjustments = metadata_adjustments.copy()
        self.palette = palette
        self.pip_info = picture_in_picture
        self.path = path

    @property
    def kelvin(self) -> Array[np.float64, ..., ...]:
        """ A property method that returns the thermogram's temperature in
            kelvin (K).

            Returns
            -------
            Array[np.float64, ..., ...]
                A 2D array of numpy float values in kelvin. Order is [H, W].
        """
        return self.__raw_to_kelvin_with_metadata(self.__thermal)

    @property
    def celsius(self) -> Array[np.float64, ..., ...]:
        """ A property method that returns the thermogram's temperature in
            degrees celsius (°C).

            Returns
            -------
            Array[np.float64, ..., ...]
                A 2D array of numpy float values in celsius. Order is [H, W].
        """
        return self.kelvin - 273.15

    @property
    def fahrenheit(self) -> Array[np.float64, ..., ...]:
        """ A property method that returns the thermogram's temperature in
            degrees fahrenheit (°F).

            Returns
            -------
            Array[np.float64, ..., ...]
                A 2D array of numpy float values in fahrenheit. Order is [H, W].
        """
        return self.celsius * 1.8 + 32.00

    @property
    def optical(self) -> Optional[Array[np.uint8, ..., ..., 3]]:
        """ Returns the thermogram's embedded photo.

            Returns
            -------
            Array[np.uint8, ..., ..., 3]
                A 3D array of 8 bit integers containing the RGB photo
                embedded within the FLIR thermogram.  Order is [H, W, C].
        """
        return None if self.__optical is None else self.__optical.copy()

    @property
    def optical_pil(self) -> Image.Image:
        """ Returns the thermogram's embedded photo as a Pillow `Image`.

            Returns
            -------
            `PIL.Image`
                A Pillow Image object of the RGB photo embedded within the FLIR
                thermogram.
        """
        return None if self.__optical is None else Image.fromarray(self.optical)

    @property
    def metadata(self) -> Dict[str, Union[float, int]]:
        metadata = self.__metadata.copy()
        metadata.update(self.__metadata_adjustments)
        return metadata

    @property
    def identifier(self) -> Optional[str]:
        return os.path.basename(self.path) if self.path is not None else None

    def render(
        self,
        min_v: Optional[float] = None,
        max_v: Optional[float] = None,
        unit: str = "kelvin",
        palette: Union[str, List[Tuple[int, int, int]]] = "embedded",
    ) -> Array[np.uint8, ..., ..., 3]:
        """ Renders the thermogram to RGB with the given settings.

            First the thermogram is normalized using the given interval and
            mode. Then the palette is used to translate the values to colors.

            Parameters
            ----------
            min_v: float or `None`. Is `None` by default.
                When set to `None`, the lower bound the FLIR camera used to render the
                original file with is used.
                    All values below this value will be clipped to this value,
                although the exact behaviour depends on the `unit`.
                    When unit is celsius, kelvin or fahrenheit, the `min_v` and `max_v`
                values function directly as the thresholds to which the thermogram is
                clipped.
                    When `unit='percentiles'`, then `min_v` and `max_v` are interpreted
                as percentiles. First the values for those percentiles are retrieved
                which are then used to clip the thermogram as described when
                `unit='kelvin'`.
            max_v: float or `None`. Is `None` by default.
                See the `min_v` for details on how it is interpreted.
            unit: str
                The unit of the `min_v` and `max_v` parameters, which can be celsius,
                fahrenheit or kelvin. Default is 'kelvin'. Only used when `mode` (see
                below) is 'minmax', thus ignored in the case of 'percentiles'.
            palette: str. Is `"embedded"` by default.
                The name of the color palette to use. See the `palettes` module to see
                which are supported in addition to "embedded".
                    Alternatively, a list of 3-tuples can be passed in. Each
                3-tuple has 3 integers corresponding to the RGB components.

            Returns
            -------
            Array[np.uint8, ..., ..., ...]
                A three dimensional array of integers between 0 and 255,
                representing an RGB render of the thermogram. Order is
                [H, W, C].
        """
        normalizer = {  # Functions defined below
            "minmax": norm.by_minmax,
            "percentiles": norm.by_percentiles,
        }

        # Validate parameters
        assert min_v is None or max_v is None or min_v < max_v
        assert not ((min_v is None or max_v is None) and unit == "percentiles")
        assert unit in ["kelvin", "celsius", "fahrenheit", "percentiles"]
        assert palette in ["grayscale", "embedded"] or palette in palettes.palettes
        if palette == "embedded" and self.palette is None:
            warnings.warn("No embedded palette detected, using grayscale instead")
            palette = "grayscale"  # Fallback in case no palette detected

        # In case the min / max values are None, find the right defaults
        emb_min_v, emb_max_v = self.embedded_range(unit)
        min_v = emb_min_v if min_v is None else min_v
        max_v = emb_max_v if max_v is None else max_v

        # Convert min/max values to kelvin if necessary
        if unit == "celsius":
            min_v = 273.15 + min_v
            max_v = 273.15 + max_v
        elif unit == "fahrenheit":
            min_v = 273.15 + (min_v - 32.0) / 1.8
            max_v = 273.15 + (max_v - 32.0) / 1.8
        mode = unit if unit == "percentiles" else "minmax"
        assert mode in normalizer.keys()

        # Render
        normalized = normalizer[mode](min_v, max_v, self.kelvin)
        if palette == "grayscale":
            # Strategy for grayscale is very different from when using a map
            rendered = (normalized * 255).astype(np.uint8)
            outshape = rendered.shape + (3,)
            repeated = np.broadcast_to(rendered[..., None], outshape)
            return np.clip(repeated, 0, 255)  # return grayscale
        elif palette == "embedded" and self.palette is not None:
            palette = self.palette.rgbs
        return palettes.map_colors(normalized, palette)  # return with color map

    def render_pil(self, edge_emphasis: float = 0.0, **kwargs) -> Image.Image:
        """ Renders the thermogram, but returns a pillow Image object.

            See `render()` for documentation on the parameters and other details.
            The parameters listed below are extra exclusive for `render_pil()`.

            Parameters
            ----------
            edge_emphasis: float
                Default is `0.0`. The strength with which to enhance the render's
                edges, using the optical image as a base. The value must be between
                `0.0` and `1.0`, where the first means no emphasis. No edge emphasis
                happens in case there is no optical image present.

            Returns
            -------
            PIL.Image
                A pillow Image of the rendered thermogram.
        """
        render_pil = Image.fromarray(self.render(**kwargs))
        if self.__optical is not None and self.pip_info and edge_emphasis > 0.0:
            ratio = self.pip_info.real_to_ir
            offset_x = self.pip_info.offset_x
            offset_y = self.pip_info.offset_y

            ratio = self.__optical.shape[1] / render_pil.size[0] / ratio
            src_size = render_pil.size
            dst_size = (
                round(render_pil.size[0] * ratio),
                round(render_pil.size[1] * ratio),
            )
            # render_pil = render_pil.resize(dst_size)

            height, width = self.__optical.shape[:2]
            origin_x, origin_y = width // 2, height // 2
            origin_x = origin_x - dst_size[0] // 2 + offset_x
            origin_y = origin_y - dst_size[1] // 2 + offset_y

            thermal_surface = (
                origin_x,
                origin_y,
                origin_x + dst_size[0],
                origin_y + dst_size[1],
            )
            embossment, opacity = self.embossment_masks_pil(opacity=edge_emphasis)
            embossment = embossment.crop(thermal_surface).resize(src_size)
            opacity = opacity.crop(thermal_surface).resize(src_size)

            render_pil = render_pil.convert("RGBA")
            render_pil.paste(embossment, (0, 0), opacity)
        return render_pil.convert("RGB")

    def picture_in_picture_pil(
        self,
        render: Optional[Image.Image] = None,
        render_opacity: float = 1.0,
        render_crop: bool = True,
        edge_emphasis: float = 0.0,
    ) -> Image.Image:
        if self.__optical is None or self.pip_info is None:
            raise ValueError("Impossible operation due missing information")

        ratio = self.pip_info.real_to_ir
        offset_x = self.pip_info.offset_x
        offset_y = self.pip_info.offset_y
        render_opacity = min(max(render_opacity, 0.0), 1.0)

        optical_pil = self.optical_pil.convert("RGBA")
        render_pil = render
        if render_pil is None:
            render_pil = self.render_pil(edge_emphasis=edge_emphasis)

        ratio = optical_pil.size[0] / render_pil.size[0] / ratio
        dst = (round(render_pil.size[0] * ratio), round(render_pil.size[1] * ratio))
        if render_crop:
            render_pil = render_pil.crop(self.pip_info.crop_box)

        dst = (round(render_pil.size[0] * ratio), round(render_pil.size[1] * ratio))
        render_pil = render_pil.resize(dst)
        render_alpha_mask = render_pil.copy()
        render_alpha_mask.putalpha(round(255 * render_opacity))

        width, height = optical_pil.size
        origin_x, origin_y = width // 2, height // 2
        origin_x = origin_x - render_pil.size[0] // 2 + offset_x
        origin_y = origin_y - render_pil.size[1] // 2 + offset_y

        optical_pil.paste(render_pil, (origin_x, origin_y), render_alpha_mask)
        return optical_pil.convert("RGB")

    def embossment_masks(self, opacity: float = 0.275) -> Tuple[np.array, np.array]:
        opacity = min(max(opacity, 0.0), 1.0)
        if self.__optical is None:
            raise ValueError("Impossible operation due to missing optical image")

        # Define azimuth, elevation, and depth
        ele = np.pi / 2.2  # radians
        azi = np.pi / 4.0  # radians
        dep = 15.0  # (0-100)

        # Get a B&W version of the image
        img = Image.fromarray(self.__optical).convert("L")
        # img = img.resize(size)
        arr_embossment = np.asarray(img).astype(float)

        # Find the gradient and adjust the gradient by the depth factor
        grad_x, grad_y = np.gradient(arr_embossment)
        grad_x = grad_x * dep / 100.0
        grad_y = grad_y * dep / 100.0

        # Get the unit incident ray
        gd = np.cos(ele)  # length of projection of ray on ground plane
        dx = gd * np.cos(azi)
        dy = gd * np.sin(azi)
        dz = np.sin(ele)

        # Find the unit normal vectors for the image
        leng = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.0)
        uni_x = grad_x / leng
        uni_y = grad_y / leng
        uni_z = 1.0 / leng

        # Take the dot product, avoiding overflow
        arr_embossment = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
        arr_embossment = arr_embossment.clip(0, 255).astype(np.uint8)

        # Threshold non-edges to be white and see-through
        arr_embossment[arr_embossment > 180] = 255
        arr_embossment = arr_embossment.astype(np.uint8)
        arr_opacity = np.full_like(arr_embossment, fill_value=round(opacity * 255))
        arr_opacity[arr_embossment > 180] = 0
        return arr_embossment, arr_opacity

    def embossment_masks_pil(self, **kwargs) -> Tuple[Image.Image, Image.Image]:
        e, o = self.embossment_masks(**kwargs)
        return Image.fromarray(e), Image.fromarray(o)

    def adjust_metadata(
        self, in_place=False, **kwargs: Union[float, int]
    ) -> "FlyrThermogram":
        """ Updates the physical metadata used to calculate the kelvin /
            celsius values based on the raw thermal data.

            This can be used to calculate kelvin/celsius values with different
            settings than the ones embedded in the thermogram itself during
            capture.

            This method does not check the given parameters. Wrong parameters
            names or values will be accepted without exceptions being raised.
            These exceptions will only occur when `kelvin` or `celsius` is
            accessed.

            Important: This does *not* adjust the metadata in the file itself;
            only the in-memory metadata used to calculate the temperatures
            returned by `kelvin` and `celsius` is updated. In other words, this
            method can *not* be used to modify or create a FLIR thermogram file
            with different camera settings.

            # Parameters
            in_place: boolean
                When False, a new `FlyrThermogram` object is returned after calling this
                method. When True, this object instance is modified in place and the
                object itself is returned.
            emissivity: float
            object_distance: float
            atmospheric_temperature: float
            ir_window_temperature: float
            ir_window_transmission: float
            reflected_apparent_temperature: float
            relative_humidity: float
            planck_r1: float
            planck_r2: float
            planck_b: float
            planck_f: int
            planck_o: int
            atmospheric_trans_alpha1: float
            atmospheric_trans_alpha2: float
            atmospheric_trans_beta1: float
            atmospheric_trans_beta2: float
            atmospheric_trans_x: float

            # Return
            FlyrThermogram
                When `in_place` is False, a new FlyrThermogram object with the updates
                settings. When `in_place` is True, the FlyrThermogram object on which this
                method is called.
        """
        msg = f"Parameter in_place incorrectly not of type bool but {type(in_place)}. Be sure to pass it first."
        assert isinstance(in_place, bool), msg
        if in_place:
            self.__metadata_adjustments.update(kwargs)
            return self

        metadata_adjustments = self.__metadata_adjustments.copy()
        metadata_adjustments.update(kwargs)

        return FlyrThermogram(
            self.__thermal.copy(),
            self.__metadata.copy(),
            None if self.__optical is None else self.__optical.copy(),
            palette=self.palette,
            picture_in_picture=self.pip_info,
            metadata_adjustments=metadata_adjustments,
            path=self.path,
        )

    def embedded_range(self, unit: str) -> Tuple[float, float]:
        range_median = self.__metadata["raw_value_median"]
        range_half = self.__metadata["raw_value_range"] / 2

        raw_vals = np.array([[range_median - range_half, range_median + range_half]])
        min_v, max_v = self.__raw_to_kelvin_with_metadata(raw_vals, orig=True).squeeze()
        if unit == "celsius":
            min_v = min_v - 273.15
            max_v = max_v - 273.15
        elif unit == "fahrenheit":
            min_v = (min_v - 273.15) * 1.8 + 32.00
            max_v = (max_v - 273.15) * 1.8 + 32.00

        return (min_v, max_v)

    def __raw_to_kelvin_with_metadata(
        self, thermal, orig: bool = False
    ) -> Array[np.float64, ..., ...]:
        metadata = self.__metadata.copy()
        if not orig:
            metadata.update(self.__metadata_adjustments)

        return FlyrThermogram.__raw_to_kelvin(
            thermal,
            metadata["emissivity"],
            metadata["object_distance"],
            metadata["atmospheric_temperature"],
            metadata["ir_window_temperature"],
            metadata["ir_window_transmission"],
            metadata["reflected_apparent_temperature"],
            metadata["relative_humidity"],
            metadata["planck_r1"],
            metadata["planck_r2"],
            metadata["planck_b"],
            metadata["planck_f"],
            metadata["planck_o"],
            metadata["atmospheric_trans_alpha1"],
            metadata["atmospheric_trans_alpha2"],
            metadata["atmospheric_trans_beta1"],
            metadata["atmospheric_trans_beta2"],
            metadata["atmospheric_trans_x"],
        )

    @staticmethod
    def __raw_to_kelvin(
        thermal,
        emissivity,
        object_distance,
        atmospheric_temperature,
        ir_window_temperature,
        ir_window_transmission,
        reflected_apparent_temperature,
        relative_humidity,
        planck_r1,
        planck_r2,
        planck_b,
        planck_f,
        planck_o,
        atmospheric_trans_alpha1,
        atmospheric_trans_alpha2,
        atmospheric_trans_beta1,
        atmospheric_trans_beta2,
        atmospheric_trans_x,
    ) -> Array[np.float64, ..., ...]:
        """ Use the details camera info metadata to translate the raw
            temperatures to °Kelvin.

            Parameters
            ----------
            thermal: Array[float, ..., ...]
                The thermal data to convert from raw values to kelvin
            emissivity : float
            object_distance : float
                Unit is meters
            atmospheric_temperature : float
                Unit is Kelvin
            ir_window_temperature : float
                Unit is Kelvin
            ir_window_transmission : float
                Unit is Kelvin
            reflected_apparent_temperature : float
                Unit is Kelvin
            relative_humidity : float
                Value in 0 and 1
            planck_r1 : float
            planck_r2 : float
            planck_b : float
            planck_f : float
            planck_o : int
            atmospheric_trans_alpha1 : float
            atmospheric_trans_alpha2 : float
            atmospheric_trans_beta1 : float
            atmospheric_trans_beta2 : float
            atmospheric_trans_x : float

            Returns
            -------
            Array[np.float64, ..., ...]
                An array of float64 values in kelvin.
        """
        # Transmission through window (calibrated)
        emiss_wind = 1 - ir_window_transmission
        refl_wind = 0

        # Transmission through the air
        water = relative_humidity * exp(
            1.5587
            + 0.06939 * (atmospheric_temperature - 273.15)
            - 0.00027816 * (atmospheric_temperature - 273.17) ** 2
            + 0.00000068455 * (atmospheric_temperature - 273.15) ** 3
        )

        def calc_atmos(alpha, beta):
            term1 = -sqrt(object_distance / 2)
            term2 = alpha + beta * sqrt(water)
            return exp(term1 * term2)

        atmos1 = calc_atmos(atmospheric_trans_alpha1, atmospheric_trans_beta1)
        atmos2 = calc_atmos(atmospheric_trans_alpha2, atmospheric_trans_beta2)
        tau1 = atmospheric_trans_x * atmos1 + (1 - atmospheric_trans_x) * atmos2
        tau2 = atmospheric_trans_x * atmos1 + (1 - atmospheric_trans_x) * atmos2

        # Radiance from the environment
        def plancked(t):
            planck_tmp = planck_r2 * (exp(planck_b / t) - planck_f)
            return planck_r1 / planck_tmp - planck_o

        raw_refl1 = plancked(reflected_apparent_temperature)
        raw_refl1_attn = (1 - emissivity) / emissivity * raw_refl1

        raw_atm1 = plancked(atmospheric_temperature)
        raw_atm1_attn = (1 - tau1) / emissivity / tau1 * raw_atm1

        term3 = emissivity * tau1 * ir_window_transmission
        raw_wind = plancked(ir_window_temperature)
        raw_wind_attn = emiss_wind / term3 * raw_wind

        raw_refl2 = plancked(reflected_apparent_temperature)
        raw_refl2_attn = refl_wind / term3 * raw_refl2

        raw_atm2 = plancked(atmospheric_temperature)
        raw_atm2_attn = (1 - tau2) / term3 / tau2 * raw_atm2

        subtraction = fsum(
            [raw_atm1_attn, raw_atm2_attn, raw_wind_attn, raw_refl1_attn, raw_refl2_attn]
        )

        raw_obj = thermal.astype(np.float64)
        raw_obj /= emissivity * tau1 * ir_window_transmission * tau2
        raw_obj -= subtraction

        # Temperature from radiance
        raw_obj += planck_o
        raw_obj *= planck_r2
        planck_term = planck_r1 / raw_obj + planck_f
        return planck_b / np.log(planck_term)
