#!/bin/bash
set -e
LMP=${LMP:-/usr/bin/lmp}
echo "== minimise T0_pristine =="
$LMP -in minimize_T0_pristine.in
echo "== minimise T1_vacancy =="
$LMP -in minimize_T1_vacancy.in
echo "== minimise T2_stone_wales =="
$LMP -in minimize_T2_stone_wales.in
echo "== minimise T3_dislocation_dipole =="
$LMP -in minimize_T3_dislocation_dipole.in
echo "== minimise T4_low_angle_gb =="
$LMP -in minimize_T4_low_angle_gb.in
echo "== minimise T5_high_angle_gb =="
$LMP -in minimize_T5_high_angle_gb.in
