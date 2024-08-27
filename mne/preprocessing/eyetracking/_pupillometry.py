# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import re

from ._hershman import based_noise_blinks_detection
from ..._fiff.constants import FIFF
from ...annotations import _annotations_starts_stops
from ...io import BaseRaw
from ...utils import _check_preload, _validate_type, logger, warn, _check_option

_KNOWN_FIND_BLINKS_METHODS = ("by_dropout", "by_slope")


def _find_dropout(data, dropout_value, sfreq):
    assert isinstance(dropout_value, (int, float)) or (isinstance(dropout_value, float) and np.isnan(dropout_value)), \
        "dropout_value must be either NaN or a numerical value (int or float)"
    assert isinstance(sfreq, (int, float)), \
        "sfreq must be numerical"
    assert np.shape(data)
    if np.isnan(dropout_value):
        dropouts_inds = np.isnan(data)
    else:
        dropouts_inds = (data == dropout_value)

    diff = np.diff(dropouts_inds.astype(int))  # Missing data onset == 1, offset == -1
    onsets = np.where(diff == 1)[0] + 1  # Blinks onset
    ends = np.where(diff == -1)[0] + 1  # Blinks offset
    # Handle case where the dropout starts from the first element
    if dropouts_inds[0]:
        onsets = np.insert(onsets, 0, 0)
    # Handle case where the dropout ends at the last element
    if dropouts_inds[-1]:
        ends = np.append(ends, len(data))
    durations = ends - onsets

    # Convert onsets and durations to seconds
    onsets = onsets / sfreq
    durations = durations / sfreq

    return onsets, durations, len(onsets)


def find_blinks(raw, method='by_dropout', per_eye=False, dropout_value=0, concat_gap_interval=100,
                description="BAD_blink"):
    """Find blinks based on the slope of the pupil size.
    
    For more information, see :footcite:`HershmanEtAl2018`.
    
    Notes
    -----
    .. versionadded:: 1.9
    """

    # Extract recorded:
    pupils_record = [ch for ch in raw.ch_names if re.search(r"pupil_", ch)]
    assert len(pupils_record) > 0, "No pupil_ channel found in raw"

    for pupil in pupils_record:
        # Check method:
        _check_option("method", method, _KNOWN_FIND_BLINKS_METHODS)
        if method == "by_dropout":
            onset, duration, n_blinks = _find_dropout(np.squeeze(raw.copy().get_data(picks=pupil)), dropout_value,
                                                      raw.info['sfreq'])
        else:
            raise Exception('Only by_dropout supported so far')
        # Add the annotations:
        raw.annotations.append(onset, duration, [description] * n_blinks,
                               ch_names=[[pupil]] * n_blinks if per_eye else None)

    return raw


def interpolate_blinks(raw, buffer=0.05, match="BAD_blink", interpolate_gaze=False):
    """Interpolate eyetracking signals during blinks.

    This function uses the timing of blink annotations to estimate missing
    data. Missing values are then interpolated linearly. Operates in place.

    Parameters
    ----------
    raw : instance of Raw
        The raw data with at least one ``'pupil'`` or ``'eyegaze'`` channel.
    buffer : float | array-like of float, shape ``(2,))``
        The time in seconds before and after a blink to consider invalid and
        include in the segment to be interpolated over. Default is ``0.05`` seconds
        (50 ms). If array-like, the first element is the time before the blink and the
        second element is the time after the blink to consider invalid, for example,
        ``(0.025, .1)``.
    match : str | list of str
        The description of annotations to interpolate over. If a list, the data within
        all annotations that match any of the strings in the list will be interpolated
        over. If a ``match`` starts with ``'BAD_'``, that part will be removed from the
        annotation description after interpolation. Defaults to ``'BAD_blink'``.
    interpolate_gaze : bool
        If False, only apply interpolation to ``'pupil channels'``. If True, interpolate
        over ``'eyegaze'`` channels as well. Defaults to False, because eye position can
        change in unpredictable ways during blinks.

    Returns
    -------
    self : instance of Raw
        Returns the modified instance.

    Notes
    -----
    .. versionadded:: 1.5
    """
    _check_preload(raw, "interpolate_blinks")
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(buffer, (float, tuple, list, np.ndarray), "buffer")
    _validate_type(match, (str, tuple, list, np.ndarray), "match")

    # determine the buffer around blinks to include in the interpolation
    buffer = np.array(buffer, dtype=float)
    if buffer.size == 1:
        buffer = np.array([buffer, buffer])

    if isinstance(match, str):
        match = [match]

    # get the blink annotations
    blink_annots = [annot for annot in raw.annotations if annot["description"] in match]
    if not blink_annots:
        warn(f"No annotations matching {match} found. Aborting.")
        return raw
    _interpolate_blinks(raw, buffer, blink_annots, interpolate_gaze=interpolate_gaze)

    # remove bad from the annotation description
    for desc in match:
        if desc.startswith("BAD_"):
            logger.info(f"Removing 'BAD_' from {desc}.")
            raw.annotations.rename({desc: desc.replace("BAD_", "")})
    return raw


def _interpolate_blinks(raw, buffer, blink_annots, interpolate_gaze):
    """Interpolate eyetracking signals during blinks in-place."""
    logger.info("Interpolating missing data during blinks...")
    pre_buffer, post_buffer = buffer
    # iterate over each eyetrack channel and interpolate the blinks
    interpolated_chs = []
    for ci, ch_info in enumerate(raw.info["chs"]):
        if interpolate_gaze:  # interpolate over all eyetrack channels
            if ch_info["kind"] != FIFF.FIFFV_EYETRACK_CH:
                continue
        else:  # interpolate over pupil channels only
            if ch_info["coil_type"] != FIFF.FIFFV_COIL_EYETRACK_PUPIL:
                continue
        # Create an empty boolean mask
        mask = np.zeros_like(raw.times, dtype=bool)
        starts, ends = _annotations_starts_stops(raw, "BAD_blink")
        starts = np.divide(starts, raw.info["sfreq"])
        ends = np.divide(ends, raw.info["sfreq"])
        for annot, start, end in zip(blink_annots, starts, ends):
            if "ch_names" not in annot or not annot["ch_names"]:
                msg = f"Blink annotation missing values for 'ch_names' key: {annot}"
                raise ValueError(msg)
            start -= pre_buffer
            end += post_buffer
            if ch_info["ch_name"] not in annot["ch_names"]:
                continue  # skip if the channel is not in the blink annotation
            # Update the mask for times within the current blink period
            mask |= (raw.times >= start) & (raw.times <= end)
        blink_indices = np.where(mask)[0]
        non_blink_indices = np.where(~mask)[0]

        # Linear interpolation
        interpolated_samples = np.interp(
            raw.times[blink_indices],
            raw.times[non_blink_indices],
            raw._data[ci, non_blink_indices],
        )
        # Replace the samples at the blink_indices with the interpolated values
        raw._data[ci, blink_indices] = interpolated_samples
        interpolated_chs.append(ch_info["ch_name"])
    if interpolated_chs:
        logger.info(
            f"Interpolated {len(interpolated_chs)} channels: {interpolated_chs}"
        )
    else:
        warn("No channels were interpolated.")
