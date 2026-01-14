from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as np
import numpy as onp
from jax import config
import dLux.utils as dlu
import equinox as eqx
import zodiax as zdx

if TYPE_CHECKING:
    from dluxshera.legacy.modeling import SheraThreePlane_Model

__all__ = [
    "BaseModeller",
    "ModelParams",
    "SheraTwoPlaneParams",
    "SheraThreePlaneParams",
    "EigenParams",
    "pack_params",
    "unpack_params",
]


class BaseModeller(zdx.Base):  # zdx.Base inherits eqx.Module
    """Base class to manage model parameters stored in a dictionary."""
    params: dict = eqx.field()  # differentiable leaves live here

    # ---------- NEW: normalization helpers ----------
    @staticmethod
    def _promote_float_like(x):
        """Promote Python/NumPy float-like values to jax arrays.
        Leave ints/bools/None unchanged. Preserve existing jax/np arrays."""
        if x is None:
            return None
        # Keep ints and bools structural/static (not differentiable)
        if isinstance(x, (bool, onp.bool_)):
            return x
        if isinstance(x, (int, onp.integer)):
            return x

        # jax/numPy arrays: preserve but ensure float arrays are jax floats
        dtype = np.float64 if config.x64_enabled else np.float32
        if isinstance(x, (np.ndarray, jax.Array)):
            # If it's already jax, keep as-is
            return x
        if isinstance(x, onp.ndarray):
            # Convert float arrays to jax float; others (e.g. int arrays) to jax arrays too if you prefer.
            if onp.issubdtype(x.dtype, onp.floating):
                return np.asarray(x, dtype=dtype)
            else:
                return np.asarray(x)

        # Python floats (or numpy floating scalars): promote to jax float arrays
        if isinstance(x, (float, onp.floating)):
            return np.array(x, dtype=dtype)

        # Lists/tuples: try to convert to a float jax array if all numeric floats
        if isinstance(x, (list, tuple)):
            try:
                arr = onp.asarray(x)
                if onp.issubdtype(arr.dtype, onp.floating):
                    return np.asarray(arr, dtype=dtype)
                # If list is ints or mixed: leave as-is (they’re usually structural, not differentiable)
                return x
            except Exception:
                return x

        # Anything else (dicts, objects, callables): leave alone
        return x

    @classmethod
    def _normalize_params_dict(cls, d: dict) -> dict:
        # Shallow normalization is enough (your params are flat)
        return {k: cls._promote_float_like(v) for k, v in d.items()}
    # ---------- /NEW ----------

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(f"Attribute {key} not found in {self.__class__.__name__}")

    def __getitem__(self, key):
        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]
        return values

    def get(self, key):
        if key in self.params:
            return self.params[key]
        raise ValueError(f"key: {key} not found in object: {type(self).__name__}")


class ModelParams(BaseModeller):
    """Encapsulates a subset of model parameters with math operations for optimization."""

    @property
    def keys(self):
        return list(self.params.keys())

    @property
    def values(self):
        return list(self.params.values())

    @property
    def grad_paths(self):
        """Return differentiable keys (kept for backwards compatibility)."""

        return list(self.params.keys())

    def replace(self, values):
        """
        Replace all parameters with the provided values.

        Parameters
        ----------
        values : dict
            A dictionary containing updated parameter values. Any parameters
            not included in this dictionary will be replaced with None.

        Returns
        ----------
        ModelParams
            A new ModelParams object with the updated parameter values.

        Notes
        ----------
        - This method expects a fully defined parameter dictionary.
        - Missing keys will be replaced with None, potentially leading
          to unexpected behavior.
        """
        normalized = self._normalize_params_dict({param: values.get(param) for param in self.keys})
        return self.set("params", normalized)

    def from_model(self, values):
        normalized = self._normalize_params_dict({param: values.get(param) for param in self.keys})
        return self.set("params", normalized)

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def __mul__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def inject(self, other):
        normalized = [self._promote_float_like(v) for v in self.values]
        return other.set(self.keys, normalized)

    def to_json(self, filepath: str):
        serializable = {
            k: (np.array(v).tolist() if v is not None else None)
            for k, v in self.params.items()
        }
        with open(filepath, "w") as f:
            json.dump(serializable, f)

    @classmethod
    def from_json(cls, filepath: str) -> "ModelParams":
        with open(filepath, "r") as f:
            raw = json.load(f)

        # Don’t collapse to Python scalars; normalize to jax where needed.
        params = {}
        for k, v in raw.items():
            params[k] = v  # leave raw shape; normalize below
        return cls(cls._normalize_params_dict(params))


class SheraThreePlaneParams(ModelParams):
    """Parameter container for the Shera Three-Plane Optical System."""

    def __init__(self, params=None, point_design=None):
        """
        Initialize the parameter set for the Shera Three-Plane Optical System.

        Parameters
        ----------
        params : dict, optional
            A dictionary of parameter overrides. These values will replace the
            default parameters, including those set by the point design.

        point_design : str, optional
            Specifies which telescope point design to use. Valid options are:
                - "shera_testbed" (default)
                - "shera_flight"

            If not specified, the default "shera_testbed" parameters will be used.
        """
        # Define the two point designs
        point_designs = {
            "shera_testbed": {
                "p1_diameter": 0.09,
                "p2_diameter": 0.025,
                "m1_focal_length": 0.35796,
                "m2_focal_length": -0.041935,
                "plane_separation": 0.320,
                "pixel_size": 6.5e-6,
                "bandwidth": 110.,  # nm
                "log_flux": np.asarray(6.78),
            },
            "shera_flight": {
                "p1_diameter": 0.22,
                "p2_diameter": 0.025,
                "m1_focal_length": 0.604353,
                "m2_focal_length": -0.0545,
                "plane_separation": 0.55413,
                "pixel_size": 4.6e-6,
                "bandwidth": 41.,  # nm
                "log_flux": np.asarray(7.13),
            }
        }

        if point_design is None:
            point_design = "shera_testbed"  # Default point design
        defaults = dict(point_designs[point_design])  # copy

        # Add other default parameters
        defaults.update({
            # Sampling and resolution settings
            "rng_seed": 0,
            "pupil_npix": 256,
            "psf_npix": 256,

            # Source parameters
            "x_position": np.asarray(0.),
            "y_position": np.asarray(0.),
            "separation": np.asarray(10.),
            "position_angle": np.asarray(90.),
            "contrast": np.asarray(0.3),
            "wavelength": 550.,  # nm
            "n_wavelengths": 3,

            # M1 Aberrations
            "m1_zernike_noll": np.arange(4, 12),
            "m1_zernike_amp": np.zeros(8),
            "m1_calibrated_power_law": 2.5,
            "m1_calibrated_amplitude": 0,
            "m1_uncalibrated_power_law": 2.5,
            "m1_uncalibrated_amplitude": 0,

            # M2 Aberrations
            "m2_zernike_noll": np.arange(4, 12),
            "m2_zernike_amp": np.zeros(8),
            "m2_calibrated_power_law": 2.5,
            "m2_calibrated_amplitude": 0,
            "m2_uncalibrated_power_law": 2.5,
            "m2_uncalibrated_amplitude": 0
        })

        # Update user-provided values
        if params is not None:
            defaults = {**defaults, **params}

        # Normalize everything to JAX arrays
        defaults = self._normalize_params_dict(defaults)

        # Initialize ModelParams with defaults
        super().__init__(defaults)

    def validate(self):
        """Validate the internal consistency of the parameter sets."""
        # Check that Zernike indexes and amplitudes match in length
        for prefix in ["m1_", "m2_"]:
            noll = self.params[f"{prefix}zernike_noll"]
            amp = self.params[f"{prefix}zernike_amp"]
            if len(noll) != len(amp):
                raise ValueError(
                    f"{prefix}zernike_noll and {prefix}zernike_amp must have the same length."
                )
        print("Validation successful.")

    def to_dict(self):
        """Flatten the parameter hierarchy for easy export."""
        return self.params

    def replace(self, values):
        """
        Replace parameters with the provided values.

        Parameters
        ----------
        values : dict
            A dictionary containing updated parameter values. Only the
            specified keys are updated, all other parameters are preserved.

        Returns
        -------
        SheraThreePlaneParams
            A new SheraThreePlaneParams object with the updated parameter values.

        Notes
        -----
        - This method preserves existing parameters that are not explicitly updated.
        - Does not support nested dictionary updates.
        """
        return self.set("params", {**self.params, **values})

    @staticmethod
    def get_param_path_map():
        '''Returns the parameter path that maps params from this class to the parameters of the model'''
        return {
            "m1_zernike_amp": "m1_aperture.coefficients",
            "m2_zernike_amp": "m2_aperture.coefficients"
        }

    def get(self, key):
        # This custom get method allows us to calculate what
        # psf_pixel_scale would be given the other parameters.
        # Falls back to the default get() behavior for other parameters
        if key == "psf_pixel_scale":
            if "psf_pixel_scale" in self.params:
                return self.params["psf_pixel_scale"]  # Return a stored value if present
            return self.compute_psf_pixel_scale()  # Otherwise compute from model geometry.

        return super().get(key)

    def update_from_model(self, model: "SheraThreePlane_Model") -> "SheraThreePlaneParams":
        """
        Return a new SheraThreePlaneParams object with updated values from the model.

        Parameters
        ----------
        model : SheraThreePlane_Model
            The model object containing updated values.

        Returns
        -------
        SheraThreePlaneParams
            A new SheraThreePlaneParams object populated with the current model parameters.
        """
        updated = {}
        param_map = self.get_param_path_map()  # Maps self keys -> model keys

        for key in self.params:
            model_key = param_map.get(key, key)
            try:
                updated_value = model.get(model_key)
            except Exception as e:
                raise ValueError(f"Could not retrieve '{model_key}' from model: {e}")
            updated[key] = updated_value

        return self.replace(updated)

    def compute_EFL(self):
        """
        Return Effective Focal Length (EFL) in m, derived from the model's optics.
        """
        # Get focal lengths and pixel size
        f1 = self.get("m1_focal_length")
        f2 = self.get("m2_focal_length")
        sep = self.get("plane_separation")

        # Effective focal length for the two-mirror relay:
        # EFL = (1/f1 + 1/f2 - sep/(f1*f2))^-1
        EFL = 1.0 / (1.0 / f1 + 1.0 / f2 - sep / (f1 * f2))  # meters
        return EFL

    def compute_psf_pixel_scale(self):
        """
        Computes the PSF pixel scale in arcseconds/pixel based on mirror geometry and pixel size.

        Returns
        -------
        float
            The computed psf_pixel_scale in arcseconds/pixel.
        """
        # Get the EFL + Pixel size
        EFL = self.compute_EFL()
        pixel_size = self.get("pixel_size")
        return np.asarray(dlu.rad2arcsec(pixel_size / EFL))  # as/pixel


class SheraTwoPlaneParams(ModelParams):
    """Parameter container for the Shera Two-Plane Optical System."""

    def __init__(self, params=None, point_design=None):
        """
        Initialize the parameter set for the Shera Two-Plane Optical System.

        Parameters
        ----------
        params : dict, optional
            A dictionary of parameter overrides. These values will replace the
            default parameters, including those set by the point design.

        point_design : str, optional
            Specifies which telescope point design to use. Valid options are:
                - "shera_testbed" (default)
                - "shera_flight"

            If not specified, the default "shera_testbed" parameters will be used.
        """
        # Define the two point designs
        point_designs = {
            "shera_testbed": {
                "p1_diameter": 0.09,
                "p2_diameter": 0.025,
                "psf_pixel_scale": 0.355,
                "bandwidth": 110.,  # nm
                "log_flux": 6.78,
            },
            "shera_flight": {
                "p1_diameter": 0.22,
                "p2_diameter": 0.025,
                "psf_pixel_scale": 0.123,
                "bandwidth": 41.,  # nm
                "log_flux": 7.13,
            }
        }

        if point_design is None:
            point_design = "shera_testbed"  # Default point design
        defaults = point_designs.get(point_design)

        # Add other default parameters
        defaults.update({
            # Sampling and resolution settings
            "rng_seed": 0,
            "pupil_npix": 256,
            "psf_npix": 256,

            # Source parameters
            "x_position": 0.,
            "y_position": 0.,
            "separation": 10.,
            "position_angle": 90.,
            "contrast": 0.3,
            "wavelength": 550.,  # nm
            "n_wavelengths": 3,

            # Aberrations
            "zernike_noll": np.arange(4, 11),
            "zernike_amp": np.zeros(7),
            "calibrated_power_law": 2.5,
            "calibrated_amplitude": 0,
            "uncalibrated_power_law": 2.5,
            "uncalibrated_amplitude": 0,
        })

        # Update user-provided values
        if params is not None:
            defaults = {**defaults, **params}

        # Initialize ModelParams with defaults
        super().__init__(defaults)

    def validate(self):
        """Validate the internal consistency of the parameter sets."""
        # Check that Zernike indexes and amplitudes match in length
        for prefix in ["m1_", "m2_"]:
            noll = self.params[f"{prefix}zernike_noll"]
            amp = self.params[f"{prefix}zernike_amp"]
            if len(noll) != len(amp):
                raise ValueError(
                    f"{prefix}zernike_noll and {prefix}zernike_amp must have the same length."
                )
        print("Validation successful.")

    def to_dict(self):
        """Flatten the parameter hierarchy for easy export."""
        return self.params

    def replace(self, values):
        """
        Replace parameters with the provided values.

        Parameters
        ----------
        values : dict
            A dictionary containing updated parameter values. Only the
            specified keys are updated, all other parameters are preserved.

        Returns
        -------
        SheraThreePlaneParams
            A new SheraThreePlaneParams object with the updated parameter values.

        Notes
        -----
        - This method preserves existing parameters that are not explicitly updated.
        - Does not support nested dictionary updates.
        """
        return self.set("params", {**self.params, **values})

    @staticmethod
    def get_param_path_map():
        '''Returns the parameter path that maps params from this class to the parameters of the model'''
        return {
            "zernike_amp": "coefficients",
        }


class EigenParams(ModelParams):
    """
    Wrapper around eigenmode coefficients, with same structure as ModelParams:
    - All differentiable leaves live in `params`, here {"eigen_coefficients": jax.Array}.
    - Metadata is static (p_ref, B, pure_keys, template).
    """
    # Differentiable leaves:
    # params: dict = eqx.field()  # inherited from ModelParams

    # <-- make arrays NON-static
    p_ref: onp.ndarray = eqx.field(static=False)
    B: onp.ndarray = eqx.field(static=False)
    # <-- keep small metadata static (hashable/comparable)
    pure_keys: list = eqx.field(static=True)  # list[str]
    shape_map: dict = eqx.field(static=True)  # {name: tuple(shape)}

    def to_pure(self):
        """Project eigen coefficients back into pure parameter vector."""
        c = self.get("eigen_coefficients")  # a jax.Array
        # p_ref, B are NumPy; that's OK because they are static. Convert as needed:
        return np.asarray(self.p_ref) + np.asarray(self.B) @ c

    def inject(self, model):
        """Project to pure parameters and inject into the model."""
        p = self.to_pure()
        d, i = {}, 0
        for name in self.pure_keys:
            shape = self.shape_map[name]
            size = 1 if not shape else int(math.prod(shape))
            d[name] = p[i:i + size].reshape(shape)
            i += size
        return ModelParams(d).inject(model)


def pack_params(values_pytree, params, model_template, from_model=False):
    """
    Flatten values into a vector + labels.
    Can handle either ModelParams/SheraThreePlaneParams or a SheraThreePlane_Model.
    """
    from dluxshera.legacy.modeling import SheraThreePlane_Model  # Importing locally avoids a circular import error
    labels = []
    flat_values = []

    path_map = model_template.get_param_path_map()
    inv_path_map = {v: k for k, v in path_map.items()}

    for param in params:
        if isinstance(values_pytree, SheraThreePlane_Model):
            # External path lookup directly from model
            value = values_pytree.get(param)
            actual_key = inv_path_map.get(param, param)
        else:
            # Internal storage key lookup from Params container
            actual_key = inv_path_map.get(param, param)
            # NOTE: some optimization containers store external keys directly
            # (e.g. "m1_aperture.coefficients") even if they are of type
            # SheraThreePlaneParams. Prefer external if present.
            if hasattr(values_pytree, "params") and (param in values_pytree.params):
                value = values_pytree.get(param)
            else:
                value = values_pytree.get(actual_key)

        if np.ndim(value) == 0:
            flat_values.append(value)
            labels.append(param)
        else:
            if actual_key == "m1_zernike_amp":
                nolls = model_template.m1_zernike_noll
                labels.extend([f"M1 Z{n}" for n in nolls])
            elif actual_key == "m2_zernike_amp":
                nolls = model_template.m2_zernike_noll
                labels.extend([f"M2 Z{n}" for n in nolls])
            else:
                labels.extend([f"{param}[{i}]" for i in range(value.size)])
            flat_values.extend(np.ravel(value))

    return np.array(flat_values), labels


def unpack_params(flat_values, params, model_template, pytree_cls=ModelParams):
    """
    Reconstruct ModelParams from a flat vector, keeping external param names.

    Parameters
    ----------
    flat_values : array-like
        Flattened parameter values (same order as pack_params).
    params : list[str]
        External optimizer parameter keys (e.g. 'm1_aperture.coefficients').
    model_template : ModelParams
        Template with shapes (e.g. SheraThreePlaneParams).
    pytree_cls : class
        Class to use for constructing the output (default=ModelParams).

    Returns
    -------
    model_params : ModelParams
        Structured parameters with external names as keys.
    """
    path_map = model_template.get_param_path_map()   # internal → external
    # Invert for convenience
    inv_path_map = {v: k for k, v in path_map.items()}

    idx = 0
    param_dict = {}

    for param in params:
        # Map external → internal only for shape lookup
        actual_key = inv_path_map.get(param, param)
        leaf = np.array(model_template.get(actual_key))
        size, shape = leaf.size, leaf.shape

        slice_vals = np.array(flat_values[idx: idx + size]).reshape(shape)
        idx += size

        # Store back under external name so it aligns with history
        param_dict[param] = slice_vals

    return pytree_cls(param_dict)
