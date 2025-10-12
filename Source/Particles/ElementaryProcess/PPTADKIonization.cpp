/* Copyright 2025
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Particles/ElementaryProcess/PPTADKIonization.H"

#include "WarpX.H"

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_IntVect.H>

#include <algorithm>
#include <array>

PPTADKIonizationFilterFunc::PPTADKIonizationFilterFunc (
    const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
    amrex::FArrayBox const& exfab,
    amrex::FArrayBox const& eyfab,
    amrex::FArrayBox const& ezfab,
    amrex::FArrayBox const& bxfab,
    amrex::FArrayBox const& byfab,
    amrex::FArrayBox const& bzfab,
    amrex::Vector<amrex::ParticleReal>& E_external_particle,
    amrex::Vector<amrex::ParticleReal>& B_external_particle,
    const amrex::Real* const AMREX_RESTRICT a_ionization_energies,
    const amrex::Real* const AMREX_RESTRICT a_adk_prefactor,
    const amrex::Real* const AMREX_RESTRICT a_adk_exp_prefactor,
    const amrex::Real* const AMREX_RESTRICT a_adk_power,
    const amrex::Real* const AMREX_RESTRICT a_adk_correction_factors,
    const amrex::Real* const AMREX_RESTRICT a_ppt_prefactor,
    const amrex::Real* const AMREX_RESTRICT a_nstar,
    const amrex::Real* const AMREX_RESTRICT a_lstar,
    amrex::Real a_laser_omega,
    int a_comp,
    int a_atomic_number,
    int a_do_adk_correction,
    int a_max_terms,
    amrex::Real a_tolerance,
    int a_offset) noexcept :
    // Initialize the embedded ADK filter - it handles all field gathering and ADK calculations
    m_adk_filter{a_pti, lev, ngEB, exfab, eyfab, ezfab, bxfab, byfab, bzfab,
                 E_external_particle, B_external_particle,
                 a_ionization_energies, a_adk_prefactor, a_adk_exp_prefactor,
                 a_adk_power, a_adk_correction_factors,
                 a_comp, a_atomic_number, a_do_adk_correction, a_offset},
    // Store PPT-specific parameters
    m_ionization_energies{a_ionization_energies},
    m_ppt_prefactor{a_ppt_prefactor},
    m_nstar{a_nstar},
    m_lstar{a_lstar},
    m_laser_omega{a_laser_omega},
    m_max_terms{a_max_terms},
    m_tolerance{a_tolerance},
    comp{a_comp},
    m_atomic_number{a_atomic_number}
{
    // No additional initialization needed - ADK filter handles everything
}
