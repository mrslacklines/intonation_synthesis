#!/bin/bash

awk 'END{printf("\"ooTextFile\"\n\"PitchTier\"\n%f %f %d\n", 0, $1, NR)}' < amu_pl_ilo_BAZA_2006D_D0100.f0 > amu_pl_ilo_BAZA_2006D_D0100.PitchTier
cat amu_pl_ilo_BAZA_2006D_D0100.f0 >> amu_pl_ilo_BAZA_2006D_D0100.PitchTier