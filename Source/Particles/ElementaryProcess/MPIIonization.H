/* Copyright 2025
 * Multiphoton Ionization Filter Function
 *
 * This file is part of WarpX.
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_MPI_IONIZATION_H_
#define WARPX_MPI_IONIZATION_H_

#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>
#include <AMReX_BaseFwd.H>

#include <cmath>

/**
 * \brief Filter function for multiphoton ionization
 *
 * Implements the multiphoton ionization rate: w = (C * E)^(2n)
 * where n is the number of photons required for ionization
 */
struct MPIIonizationFilterFunc
{
    // Ionization data
    const amrex::Real* AMREX_RESTRICT m_ionization_energies;
    const int* AMREX_RESTRICT m_photon_numbers;         // n for each level
    const amrex::Real* AMREX_RESTRICT m_rate_prefactor; // (dt / t_au) * sigma^(2n), sigma = C * E_au

    int comp;  // ionizationLevel component index
    int m_atomic_number;

    // Field gathering machinery (same as ADK)
    GetParticlePosition<PIdx> m_get_position;
    GetExternalEBField m_get_externalEB;
    amrex::ParticleReal m_Ex_external_particle;
    amrex::ParticleReal m_Ey_external_particle;
    amrex::ParticleReal m_Ez_external_particle;
    amrex::ParticleReal m_Bx_external_particle;
    amrex::ParticleReal m_By_external_particle;
    amrex::ParticleReal m_Bz_external_particle;

    amrex::Array4<const amrex::Real> m_ex_arr;
    amrex::Array4<const amrex::Real> m_ey_arr;
    amrex::Array4<const amrex::Real> m_ez_arr;
    amrex::Array4<const amrex::Real> m_bx_arr;
    amrex::Array4<const amrex::Real> m_by_arr;
    amrex::Array4<const amrex::Real> m_bz_arr;

    amrex::IndexType m_ex_type;
    amrex::IndexType m_ey_type;
    amrex::IndexType m_ez_type;
    amrex::IndexType m_bx_type;
    amrex::IndexType m_by_type;
    amrex::IndexType m_bz_type;

    amrex::XDim3 m_dinv;
    amrex::XDim3 m_xyzmin;

    bool m_galerkin_interpolation;
    int m_nox;
    int m_n_rz_azimuthal_modes;

    amrex::Dim3 m_lo;

    /**
     * \brief Constructor - sets up field gathering and stores MPI coefficients
     */
    MPIIonizationFilterFunc (const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
                             amrex::FArrayBox const& exfab,
                             amrex::FArrayBox const& eyfab,
                             amrex::FArrayBox const& ezfab,
                             amrex::FArrayBox const& bxfab,
                             amrex::FArrayBox const& byfab,
                             amrex::FArrayBox const& bzfab,
                             amrex::Vector<amrex::ParticleReal>& E_external_particle,
                             amrex::Vector<amrex::ParticleReal>& B_external_particle,
                             const amrex::Real* const AMREX_RESTRICT a_ionization_energies,
                             const int* const AMREX_RESTRICT a_photon_numbers,
                             const amrex::Real* const AMREX_RESTRICT a_rate_prefactor,
                             int a_comp,
                             int a_atomic_number,
                             int a_offset = 0) noexcept;

    /**
     * \brief Operator - evaluates MPI ionization probability for each particle
     *
     * Computes the multiphoton ionization rate w = (C*E)^(2n) and stochastically
     * determines if ionization occurs based on the probability p = 1 - exp(-w*dt/gamma)
     */
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PData& ptd, int i, amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex::literals;

        const int ion_lev = ptd.m_runtime_idata[comp][i];
        if (ion_lev >= m_atomic_number) { return false; }

        constexpr amrex::Real c = PhysConst::c;
        constexpr amrex::Real c2_inv = amrex::Real(1.)/c/c;

        // Get particle position
        amrex::ParticleReal xp, yp, zp;
        m_get_position(i, xp, yp, zp);

        // Initialize with external fields
        amrex::ParticleReal ex = m_Ex_external_particle;
        amrex::ParticleReal ey = m_Ey_external_particle;
        amrex::ParticleReal ez = m_Ez_external_particle;
        amrex::ParticleReal bx = m_Bx_external_particle;
        amrex::ParticleReal by = m_By_external_particle;
        amrex::ParticleReal bz = m_Bz_external_particle;

        m_get_externalEB(i, ex, ey, ez, bx, by, bz);

        // Gather E and B fields from grid
        doGatherShapeN(xp, yp, zp, ex, ey, ez, bx, by, bz,
                       m_ex_arr, m_ey_arr, m_ez_arr,
                       m_bx_arr, m_by_arr, m_bz_arr,
                       m_ex_type, m_ey_type, m_ez_type,
                       m_bx_type, m_by_type, m_bz_type,
                       m_dinv, m_xyzmin, m_lo, m_n_rz_azimuthal_modes,
                       m_nox, m_galerkin_interpolation);

        // Get particle momentum
        const amrex::ParticleReal ux = ptd.m_rdata[PIdx::ux][i];
        const amrex::ParticleReal uy = ptd.m_rdata[PIdx::uy][i];
        const amrex::ParticleReal uz = ptd.m_rdata[PIdx::uz][i];

        // Compute Lorentz factor
        const auto gamma = static_cast<amrex::Real>(
            std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz) * c2_inv));

        // Compute electric field amplitude in particle rest frame
        // Same formula as ADK ionization (relativistic field transformation)
        const amrex::Real E = std::sqrt(
            - (ux*ex + uy*ey + uz*ez) * (ux*ex + uy*ey + uz*ez) * c2_inv
            + (gamma*ex + uy*bz - uz*by) * (gamma*ex + uy*bz - uz*by)
            + (gamma*ey + uz*bx - ux*bz) * (gamma*ey + uz*bx - ux*bz)
            + (gamma*ez + ux*by - uy*bx) * (gamma*ez + ux*by - uy*bx)
        );

        // Compute multiphoton ionization rate in atomic units for performance
        // Atomic unit of electric field: E_au = 5.14220674763e11 V/m
        // In atomic units: w = E_au^(2n) per atomic timescale
        constexpr amrex::Real E_atomic = PhysConst::E_au;  // V/m
        const amrex::Real E_norm = E / E_atomic;  // normalized electric field
        
        const int two_n = 2 * m_photon_numbers[ion_lev];
        
        // Fast integer power using repeated squaring (avoids slow std::pow)
        // For two_n = 16: E^16 = ((E^2)^2)^2)^2 = 4 multiplications vs 15 for naive approach
        amrex::Real E_power = 1.0_rt;
        amrex::Real base = E_norm;
        int exp = two_n;
        
        // Binary exponentiation algorithm
        while (exp > 0) {
            if (exp & 1) {  // if exp is odd
                E_power *= base;
            }
            base *= base;   // square the base
            exp >>= 1;      // divide exponent by 2
        }
        
        // Apply rate prefactor and Lorentz factor
        // rate_prefactor = (dt/t_au) * sigma^(2n), where sigma = C * E_au
        // w_dtau = rate_prefactor * E_norm^(2n) / gamma = (dt/t_au) * (sigma * E_norm)^(2n) / gamma
        const amrex::Real w_dtau = m_rate_prefactor[ion_lev] * E_power / gamma;

        // Convert rate to probability
        // For small w_dtau: p ≈ w_dtau
        // For large w_dtau: p = 1 - exp(-w_dtau)
        const amrex::Real p = 1._rt - std::exp(-w_dtau);

        // Monte Carlo decision
        const amrex::Real random_draw = amrex::Random(engine);
        return (random_draw < p);
    }
};

#endif // WARPX_MPI_IONIZATION_H_
