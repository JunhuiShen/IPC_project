"""Clearance used by the broad phase for FP32 linear CCD queries."""

from ctypes import c_float

# Use the same represented distance as LinearQuery::gap_floor in ccd_fp32.cuh.
LINEAR_CCD_GAP_FLOOR = c_float(1e-7).value

# Clearance for moving-point/fixed-triangle queries in the motion projection experiment.
MOTION_PROJECTION_POINT_GAP = 1e-8
