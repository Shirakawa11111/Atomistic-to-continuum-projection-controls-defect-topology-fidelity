#!/bin/bash
set -e
LMP=${LMP:-/usr/bin/lmp}
echo "== MD T0_pristine =="
$LMP -in md_T0_pristine.in
echo "== MD T1_vacancy =="
$LMP -in md_T1_vacancy.in
echo "== MD T2_stone_wales =="
$LMP -in md_T2_stone_wales.in
echo "== MD T3_dislocation_dipole =="
$LMP -in md_T3_dislocation_dipole.in
echo "== MD T4_low_angle_gb =="
$LMP -in md_T4_low_angle_gb.in
echo "== MD T5_high_angle_gb =="
$LMP -in md_T5_high_angle_gb.in
