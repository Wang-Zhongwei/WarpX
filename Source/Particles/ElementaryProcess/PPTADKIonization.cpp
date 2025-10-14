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
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuUtility.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>

amrex::Gpu::DeviceVector<amrex::Real>
PPTADKIonizationFilterFunc::WmLookupTable::s_values;

amrex::Gpu::DeviceScalar<amrex::Real*>
PPTADKIonizationFilterFunc::WmLookupTable::s_values_ptr;

amrex::Gpu::DeviceScalar<int>
PPTADKIonizationFilterFunc::WmLookupTable::s_ready_flag(0);

std::once_flag PPTADKIonizationFilterFunc::WmLookupTable::s_init_once;

void
PPTADKIonizationFilterFunc::WmLookupTable::Initialize () noexcept
{
    std::call_once(s_init_once, []()
    {
        const int table_size = (max_cached_m + 1) * num_points;
        amrex::Vector<amrex::Real> h_values(table_size, 0._rt);

        for (int m = 0; m <= max_cached_m; ++m) {
            for (int i = 0; i < num_points; ++i) {
                const amrex::Real x = static_cast<amrex::Real>(i) * dx;
                h_values[m * num_points + i] =
                    PPTADKIonizationFilterFunc::compute_w_m_integral(x, m);
            }
        }

        s_values.resize(table_size);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                              h_values.begin(), h_values.end(),
                              s_values.begin());

        s_values_ptr.setValue(s_values.data());
        s_ready_flag.setValue(1);
        amrex::Gpu::streamSynchronize();
    });
}

AMREX_GPU_HOST_DEVICE
bool
PPTADKIonizationFilterFunc::WmLookupTable::IsReady () noexcept
{
    return *(s_ready_flag.data()) != 0;
}

AMREX_GPU_HOST_DEVICE
amrex::Real
PPTADKIonizationFilterFunc::WmLookupTable::Lookup (int mabs, amrex::Real xabs) noexcept
{
    const amrex::Real clamped_x = amrex::min(xabs, x_max);
    const amrex::Real scaled = clamped_x * inv_dx;
    int idx = static_cast<int>(scaled);
    if (idx >= num_points - 1) {
        idx = num_points - 2;
    }

    const amrex::Real frac = scaled - static_cast<amrex::Real>(idx);
    const amrex::Real* values = *(s_values_ptr.data());
    const int offset = mabs * num_points + idx;
    const amrex::Real y0 = values[offset];
    const amrex::Real y1 = values[offset + 1];
    return y0 + frac * (y1 - y0);
}

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
    WmLookupTable::Initialize();
    // No additional initialization needed - ADK filter handles everything
}
