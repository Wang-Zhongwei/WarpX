# Multiphoton Ionization Implementation Plan for WarpX

## Overview

This document outlines the implementation plan for adding multiphoton ionization (MPI) capability to WarpX. The implementation will follow a similar structure to the existing ADK field ionization, allowing species to be ionized via multiphoton absorption.

### Physical Model

**Multiphoton Ionization Rate:**
```
w = (C * E)^(2n)
```

Where:
- `E` is the electric field amplitude in the particle's rest frame
- `n` is the multiphoton order = ceil(I_p / (ℏω))
- `I_p` is the ionization potential (from existing table)
- `ω = 2πc/λ₀` is the laser angular frequency
- `λ₀` is the laser wavelength (initially hardcoded as 800 nm)
- `C` is a species-dependent prefactor (default: 1e-14)

### Initial Species Support

From the provided data table, initial support for:
- H⁺: I_p = 13.6 eV → n(800nm) = 9
- C¹⁺: I_p = 11.26 eV → n(800nm) = 8  
- C²⁺: I_p = 24.38 eV → n(800nm) = 16
- C₆H₆⁺: I_p = 9.24 eV → n(800nm) = 6

## Implementation Strategy

The implementation will be divided into **TWO phases** for efficient delegation:

### Phase 1: Core Multiphoton Module (Data Structures & Initialization)
**Goal:** Add all data structures, input parsing, and initialization without modifying execution logic

### Phase 2: Execution & Integration (Runtime Calculation)
**Goal:** Implement the multiphoton filter function and integrate into the time-stepping loop

---

## Phase 1: Core Multiphoton Module (Data Structures & Initialization)

### 1.1 Input Parameter Parsing

**File:** `Source/Particles/PhysicalParticleContainer.cpp`

**Add to `PhysicalParticleContainer::InitIonizationModule()` (after line ~131):**

```cpp
// Determine ionization model (ADK or MPI)
std::string ionization_model = "ADK";  // default
utils::parser::queryWithParser(pp_species_name, "ionization_model", ionization_model);

if (ionization_model == "multiphoton" || ionization_model == "MPI") {
    do_mpi_ionization = 1;
    do_field_ionization = 0;  // MPI uses different code path

    // Parse MPI-specific parameters
    amrex::Real mpi_lambda_nm = 800.0;  // default 800 nm
    utils::parser::queryWithParser(pp_species_name, "mpi_laser_wavelength", mpi_lambda_nm);

    // Calculate omega from lambda
    constexpr amrex::Real c_SI = PhysConst::c;  // m/s
    mpi_laser_omega = 2.0 * MathConst::pi * c_SI / (mpi_lambda_nm * 1.0e-9);

    // Optional: MPI prefactor C (default = 1e-14, can be species-specific)
    mpi_prefactor_C = 1.0e-14;
    utils::parser::queryWithParser(pp_species_name, "mpi_prefactor", mpi_prefactor_C);
}
```

**File:** `Source/Particles/WarpXParticleContainer.H`

**Add member variables (around line ~630, near ionization members):**

```cpp
// Multiphoton ionization (MPI) parameters
int do_mpi_ionization = 0;
amrex::Real mpi_laser_omega = 0.0;      // angular frequency (rad/s)
amrex::Real mpi_prefactor_C = 1.0e-14;  // prefactor in w = (C*E)^(2n)
amrex::Gpu::DeviceVector<int> mpi_photon_numbers;     // n for each level
amrex::Gpu::DeviceVector<amrex::Real> mpi_rates_prefactor;  // dt * C^(2n) for each level
```

### 1.2 MPI Coefficient Precomputation

**File:** `Source/Particles/PhysicalParticleContainer.cpp`

**Add to `PhysicalParticleContainer::InitIonizationModule()` (after ionization energies are loaded):**

```cpp
if (do_mpi_ionization) {
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        mpi_laser_omega > 0.0,
        "MPI requires valid laser wavelength (mpi_laser_wavelength parameter)");
    
    // Allocate MPI arrays
    mpi_photon_numbers.resize(ion_atomic_number);
    mpi_rates_prefactor.resize(ion_atomic_number);
    
    const Real dt = WarpX::GetInstance().getdt(0);
    const Real hbar_SI = PhysConst::hbar;  // J·s
    const Real eV_to_J = PhysConst::q_e;   // conversion factor
    const Real photon_energy_eV = hbar_SI * mpi_laser_omega / eV_to_J;
    
    Real const* AMREX_RESTRICT p_ionization_energies = ionization_energies.data();
    int* AMREX_RESTRICT p_mpi_n = mpi_photon_numbers.data();
    Real* AMREX_RESTRICT p_mpi_prefactor = mpi_rates_prefactor.data();
    
    const Real C = mpi_prefactor_C;
    
    amrex::ParallelFor(ion_atomic_number, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        const Real Ip_eV = p_ionization_energies[i];
        // Calculate multiphoton order
        p_mpi_n[i] = static_cast<int>(std::ceil(Ip_eV / photon_energy_eV));

        // Precompute rate prefactor: dt * C^(2n)
        // Units: C should be in SI units (appropriate for E in V/m)
        const int two_n = 2 * p_mpi_n[i];
        p_mpi_prefactor[i] = dt * std::pow(C, static_cast<Real>(two_n));
    });
    
    Gpu::synchronize();
    
    // Print MPI initialization info
    amrex::Print() << "Multiphoton ionization initialized for species '" 
                   << species_name << "':\n"
                   << "  Laser wavelength: " << (2.0*MathConst::pi*c_SI/mpi_laser_omega)*1e9 
                   << " nm\n"
                   << "  Photon energy: " << photon_energy_eV << " eV\n"
                   << "  Prefactor C: " << mpi_prefactor_C << "\n";
}
```

### 1.3 Function Declarations

**File:** `Source/Particles/PhysicalParticleContainer.H`

**Add method declaration (around line ~186, near getIonizationFunc):**

```cpp
/**
 * \brief Create multiphoton ionization filter function
 */
MPIIonizationFilterFunc getMPIIonizationFunc (const WarpXParIter& pti,
                                               int lev,
                                               amrex::IntVect ngEB,
                                               const amrex::FArrayBox& Ex,
                                               const amrex::FArrayBox& Ey,
                                               const amrex::FArrayBox& Ez,
                                               const amrex::FArrayBox& Bx,
                                               const amrex::FArrayBox& By,
                                               const amrex::FArrayBox& Bz);
```

### 1.4 Create New Header File for MPI Filter Function

**File:** `Source/Particles/ElementaryProcess/MPIIonization.H` (NEW FILE)

```cpp
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
    const amrex::Real* AMREX_RESTRICT m_rate_prefactor; // dt * C^n
    
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
    
    // Constructor (to be implemented in Phase 2)
    MPIIonizationFilterFunc (const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
                             amrex::FArrayBox const& exfab,
                             amrex::FArrayBox const& eyfab,
                             amrex::FArrayBox const& ezfab,
                             amrex::FArrayBox const& bxfab,
                             amrex::FArrayBox const& byfab,
                             amrex::FArrayBox const& bzfab,
                             amrex::Vector<amrex::ParticleReal>& E_external_particle,
                             amrex::Vector<amrex::ParticleReal>& B_external_particle,
                             const amrex::Real* AMREX_RESTRICT a_ionization_energies,
                             const int* AMREX_RESTRICT a_photon_numbers,
                             const amrex::Real* AMREX_RESTRICT a_rate_prefactor,
                             int a_comp,
                             int a_atomic_number,
                             int a_offset = 0) noexcept;
    
    // Operator (to be implemented in Phase 2)
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PData& ptd, int i, amrex::RandomEngine const& engine) const noexcept;
};

#endif // WARPX_MPI_IONIZATION_H_
```

### 1.5 Update Documentation Stub

**File:** `Docs/source/theory/multiphysics/ionization.rst`

**Add section (at end):**

```rst

Multiphoton Ionization (MPI)
-----------------------------

WarpX also supports multiphoton ionization, where atoms/ions absorb multiple photons
simultaneously to reach ionization. The ionization rate is:

.. math::

   w = (C \cdot E)^{2n}

where:

* :math:`E` is the electric field amplitude in the particle rest frame
* :math:`n = \lceil I_p / (\hbar\omega) \rceil` is the minimum number of photons required
* :math:`I_p` is the ionization potential
* :math:`\omega` is the laser angular frequency
* :math:`C` is a species-dependent prefactor (default: 1e-14)

To enable MPI for a species::

    species_name.ionization_model = "multiphoton"
    species_name.mpi_laser_wavelength = 800.0  # nm
    species_name.mpi_prefactor = 1.0e-14  # optional
```

---

## Phase 2: Execution & Integration (Runtime Calculation)

### 2.1 Implement MPI Filter Constructor

**File:** `Source/Particles/ElementaryProcess/MPIIonization.cpp` (NEW FILE)

```cpp
#include "MPIIonization.H"
#include "WarpX.H"

MPIIonizationFilterFunc::MPIIonizationFilterFunc (
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
    const int* const AMREX_RESTRICT a_photon_numbers,
    const amrex::Real* const AMREX_RESTRICT a_rate_prefactor,
    int a_comp,
    int a_atomic_number,
    int a_offset) noexcept :
    m_ionization_energies{a_ionization_energies},
    m_photon_numbers{a_photon_numbers},
    m_rate_prefactor{a_rate_prefactor},
    comp{a_comp},
    m_atomic_number{a_atomic_number},
    m_Ex_external_particle{E_external_particle[0]},
    m_Ey_external_particle{E_external_particle[1]},
    m_Ez_external_particle{E_external_particle[2]},
    m_Bx_external_particle{B_external_particle[0]},
    m_By_external_particle{B_external_particle[1]},
    m_Bz_external_particle{B_external_particle[2]},
    m_galerkin_interpolation{WarpX::galerkin_interpolation},
    m_nox{WarpX::nox},
    m_n_rz_azimuthal_modes{WarpX::n_rz_azimuthal_modes}
{
    using namespace amrex::literals;
    
    m_get_position = GetParticlePosition<PIdx>(a_pti, a_offset);
    m_get_externalEB = GetExternalEBField(a_pti, a_offset);
    
    m_ex_arr = exfab.array();
    m_ey_arr = eyfab.array();
    m_ez_arr = ezfab.array();
    m_bx_arr = bxfab.array();
    m_by_arr = byfab.array();
    m_bz_arr = bzfab.array();
    
    m_ex_type = exfab.box().ixType();
    m_ey_type = eyfab.box().ixType();
    m_ez_type = ezfab.box().ixType();
    m_bx_type = bxfab.box().ixType();
    m_by_type = byfab.box().ixType();
    m_bz_type = bzfab.box().ixType();
    
    amrex::Box box = a_pti.tilebox();
    box.grow(ngEB);
    
    const std::array<amrex::Real,3>& dx = WarpX::CellSize(std::max(lev, 0));
    m_dinv = amrex::XDim3{1._rt/dx[0], 1._rt/dx[1], 1._rt/dx[2]};
    
    m_xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
    m_lo = amrex::lbound(box);
}
```

### 2.2 Implement MPI Filter Operator

**File:** `Source/Particles/ElementaryProcess/MPIIonization.H`

**Add template implementation after the struct definition:**

```cpp
template <typename PData>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool MPIIonizationFilterFunc::operator() (
    const PData& ptd, int i, amrex::RandomEngine const& engine) const noexcept
{
    using namespace amrex::literals;
    
    const int ion_lev = ptd.m_runtime_idata[comp][i];
    if (ion_lev >= m_atomic_number) { return false; }
    
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
    constexpr amrex::Real c = PhysConst::c;
    constexpr amrex::Real c2_inv = 1._rt / (c * c);
    const auto gamma = static_cast<amrex::Real>(
        std::sqrt(1._rt + (ux*ux + uy*uy + uz*uz) * c2_inv));
    
    // Compute electric field amplitude in particle rest frame
    // Same formula as ADK ionization
    const amrex::Real E = std::sqrt(
        - (ux*ex + uy*ey + uz*ez) * (ux*ex + uy*ey + uz*ez) * c2_inv
        + (gamma*ex + uy*bz - uz*by) * (gamma*ex + uy*bz - uz*by)
        + (gamma*ey + uz*bx - ux*bz) * (gamma*ey + uz*bx - ux*bz)
        + (gamma*ez + ux*by - uy*bx) * (gamma*ez + ux*by - uy*bx)
    );
    
    // Compute multiphoton ionization rate: w = (C*E)^(2n)
    // In proper time: w*dtau = w*dt/gamma
    const int n = m_photon_numbers[ion_lev];
    const int two_n = 2 * n;
    const amrex::Real rate_prefactor = m_rate_prefactor[ion_lev];  // dt * C^(2n)

    amrex::Real w_dtau = 0._rt;
    if (E > 0._rt) {
        // w_dtau = (dt * C^(2n)) * E^(2n) / gamma = rate_prefactor * E^(2n) / gamma
        w_dtau = rate_prefactor * std::pow(E, static_cast<amrex::Real>(two_n)) / gamma;
    }
    
    // Convert rate to probability
    // For small w_dtau: p ≈ w_dtau
    // For large w_dtau: p = 1 - exp(-w_dtau)
    const amrex::Real p = 1._rt - std::exp(-w_dtau);
    
    // Monte Carlo decision
    const amrex::Real random_draw = amrex::Random(engine);
    return (random_draw < p);
}
```

### 2.3 Implement getMPIIonizationFunc

**File:** `Source/Particles/PhysicalParticleContainer.cpp`

**Add new function (near getIonizationFunc around line ~1552):**

```cpp
MPIIonizationFilterFunc
PhysicalParticleContainer::getMPIIonizationFunc (
    const WarpXParIter& pti,
    int lev,
    amrex::IntVect ngEB,
    const amrex::FArrayBox& Ex,
    const amrex::FArrayBox& Ey,
    const amrex::FArrayBox& Ez,
    const amrex::FArrayBox& Bx,
    const amrex::FArrayBox& By,
    const amrex::FArrayBox& Bz)
{
    WARPX_PROFILE("PhysicalParticleContainer::getMPIIonizationFunc()");
    
    return {pti, lev, ngEB, Ex, Ey, Ez, Bx, By, Bz,
            m_E_external_particle, m_B_external_particle,
            ionization_energies.dataPtr(),
            mpi_photon_numbers.dataPtr(),
            mpi_rates_prefactor.dataPtr(),
            GetIntCompIndex("ionizationLevel"),
            ion_atomic_number};
}
```

### 2.4 Integrate into doFieldIonization Loop

**File:** `Source/Particles/MultiParticleContainer.cpp`

**Modify `doFieldIonization()` (around line ~267-305):**

```cpp
// Loop over all species.
// Ionized particles in pc_source create particles in pc_product
for (auto& pc_source : allcontainers)
{
    // Skip if neither ADK nor MPI ionization enabled
    if (!pc_source->do_field_ionization && !pc_source->do_mpi_ionization) { 
        continue; 
    }
    
    auto& pc_product = allcontainers[pc_source->ionization_product];
    
    const SmartCopyFactory copy_factory(*pc_source, *pc_product);
    auto *phys_pc_ptr = static_cast<PhysicalParticleContainer*>(pc_source.get());
    
    auto Copy      = copy_factory.getSmartCopy();
    auto Transform = IonizationTransformFunc();
    
    pc_source ->defineAllParticleTiles();
    pc_product->defineAllParticleTiles();
    
    auto info = getMFItInfo(*pc_source, *pc_product);
    
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter pti(*pc_source, lev, info); pti.isValid(); ++pti)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());
        
        auto& src_tile = pc_source ->ParticlesAt(lev, pti);
        auto& dst_tile = pc_product->ParticlesAt(lev, pti);
        
        // Get appropriate filter function based on ionization model
        if (pc_source->do_field_ionization) {
            // ADK ionization (existing code)
            auto Filter = phys_pc_ptr->getIonizationFunc(
                pti, lev, Ex.nGrowVect(),
                Ex[pti], Ey[pti], Ez[pti],
                Bx[pti], By[pti], Bz[pti]);
            
            const auto np_dst = dst_tile.numParticles();
            const auto num_added = filterCopyTransformParticles<1>(
                *pc_product, dst_tile, src_tile, np_dst,
                Filter, Copy, Transform);
            
            setNewParticleIDs(dst_tile, np_dst, num_added);
        } 
        else if (pc_source->do_mpi_ionization) {
            // Multiphoton ionization (new code)
            auto Filter = phys_pc_ptr->getMPIIonizationFunc(
                pti, lev, Ex.nGrowVect(),
                Ex[pti], Ey[pti], Ez[pti],
                Bx[pti], By[pti], Bz[pti]);
            
            const auto np_dst = dst_tile.numParticles();
            const auto num_added = filterCopyTransformParticles<1>(
                *pc_product, dst_tile, src_tile, np_dst,
                Filter, Copy, Transform);
            
            setNewParticleIDs(dst_tile, np_dst, num_added);
        }
        
        // [rest of timing/cost tracking code unchanged]
```

### 2.5 Add Include Directives

**File:** `Source/Particles/PhysicalParticleContainer.H`

Add near top (around line ~10-20):
```cpp
#include "Particles/ElementaryProcess/MPIIonization.H"
```

**File:** `Source/Particles/MultiParticleContainer.cpp`

Add near top:
```cpp
#include "Particles/ElementaryProcess/MPIIonization.H"
```

---

## Testing & Validation

### Phase 1 Testing

After Phase 1 completion, verify:

1. **Compilation Test:**
   ```bash
   cd /users/PAS2137/wang15032/src/warpx
   
   # Clean previous build
   rm -rf build_test_mpi
   
   # Configure with CMake
   cmake -S . -B build_test_mpi \
     -DWarpX_DIMS=3 \
     -DWarpX_COMPUTE=OMP \
     -DWarpX_MPI=ON \
     -DWarpX_OPENPMD=ON \
     -DWarpX_QED=ON
   
   # Build (should compile without errors)
   cmake --build build_test_mpi -j 8
   ```

2. **Parameter Parsing Test:**
   Create a test input file `test_mpi_init.txt`:
   ```
   max_step = 0  # Exit after initialization
   
   particles.species_names = electrons ions
   
   ions.species_type = hydrogen
   ions.injection_style = nuniformpercell
   ions.num_particles_per_cell_each_dim = 1 1 1
   ions.do_field_ionization = 0
   ions.do_mpi_ionization = 1
   ions.ionization_model = "multiphoton"
   ions.mpi_laser_wavelength = 800.0
   ions.mpi_prefactor = 1.0
   ions.physical_element = H
   ions.ionization_initial_level = 0
   ions.ionization_product_species = electrons
   ions.charge = q_e
   
   electrons.species_type = electron
   electrons.injection_style = nuniformpercell
   electrons.num_particles_per_cell_each_dim = 0 0 0
   
   # [minimal grid setup]
   amr.n_cell = 8 8 8
   amr.max_level = 0
   geometry.dims = 3
   geometry.prob_lo = -1.e-6 -1.e-6 -1.e-6
   geometry.prob_hi =  1.e-6  1.e-6  1.e-6
   ```
   
   Run:
   ```bash
   ./build_test_mpi/bin/warpx.3d test_mpi_init.txt
   ```
   
   Expected output should include:
   ```
   Multiphoton ionization initialized for species 'ions':
     Laser wavelength: 800 nm
     Photon energy: 1.549 eV
     Prefactor C: 1.0
   ```

### Phase 2 Testing

After Phase 2 completion, verify:

1. **Full Execution Test:**
   Create `test_mpi_execution.txt` with actual fields and particles:
   ```
   max_step = 10
   warpx.const_dt = 1.0e-16
   
   # Add laser or external E-field
   particles.E_ext_particle_init_style = "constant"
   particles.E_external_particle = 1.0e11 0.0 0.0  # V/m
   
   # [rest similar to init test]
   ```
   
   Run and check:
   ```bash
   ./build_test_mpi/bin/warpx.3d test_mpi_execution.txt
   ```
   
   - Should complete without crashes
   - Check that electrons are created (ionization occurring)
   - Verify ionizationLevel increments properly

2. **Comparison Test (Optional):**
   Compare MPI vs ADK ionization rates for hydrogen in known field conditions.

3. **Unit Test (Optional):**
   - Test n calculation for known I_p values
   - Verify E^n computation doesn't overflow
   - Check probability calculation

### Build System Integration

**File:** `Source/Particles/ElementaryProcess/Make.package`

Add the new source file:
```make
CEXE_sources += MPIIonization.cpp
```

For CMake, the new `.cpp` file should be automatically picked up if placed in the correct directory.

---

## Key Design Decisions & Notes

### 1. Separate vs. Unified Code Path

**Decision:** Keep MPI separate from ADK (`do_mpi_ionization` flag) rather than unified.

**Rationale:**
- Cleaner code separation during development
- Different physics (power law vs exponential)
- Can be unified later if desired

### 2. Field Calculation Reuse

**Decision:** Reuse exact same E-field calculation as ADK ionization.

**Rationale:**
- Both need E in particle rest frame
- Already correct for boosted frames
- Leverages existing, tested code

### 3. Prefactor C Units

**Decision:** C should have units such that (C*E)^(2n) has units of rate (1/s).

**Rationale:**
- If E is in V/m = kg·m/(A·s³), and we want w in 1/s
- For w = (C*E)^(2n) in 1/s: [C] = (s)^(1/(2n)) · (A·s³/(kg·m))
- User must provide C in SI units (default: 1e-14)

**Alternative:** Could absorb units into E conversion, but less transparent.

### 4. Photon Number Calculation

**Decision:** Use `ceil()` rather than `floor()` for photon number.

**Rationale:**
- Need at least n photons to exceed I_p
- Conservative approach ensures energy threshold met
- Can be made configurable if needed

### 5. Data Structure Reuse

**Decision:** Reuse existing `ionization_energies`, `ionization_product`, `ionizationLevel`.

**Rationale:**
- Same physics of ionization state tracking
- Avoids duplication
- Works seamlessly with existing deposition code scaling by ionizationLevel

---

## File Checklist

### Phase 1 Files to Modify:
- [ ] `Source/Particles/WarpXParticleContainer.H` - Add member variables
- [ ] `Source/Particles/PhysicalParticleContainer.H` - Add function declaration
- [ ] `Source/Particles/PhysicalParticleContainer.cpp` - Parse params, precompute coefficients
- [ ] `Docs/source/theory/multiphysics/ionization.rst` - Add documentation

### Phase 1 Files to Create:
- [ ] `Source/Particles/ElementaryProcess/MPIIonization.H` - Filter struct definition

### Phase 2 Files to Modify:
- [ ] `Source/Particles/ElementaryProcess/MPIIonization.H` - Implement operator
- [ ] `Source/Particles/PhysicalParticleContainer.cpp` - Implement getMPIIonizationFunc
- [ ] `Source/Particles/MultiParticleContainer.cpp` - Integrate into loop
- [ ] `Source/Particles/ElementaryProcess/Make.package` - Add to build

### Phase 2 Files to Create:
- [ ] `Source/Particles/ElementaryProcess/MPIIonization.cpp` - Constructor implementation

---

## Post-Implementation Tasks

After both phases complete:

1. **Extended Testing:**
   - Test with H, C1+, C2+ from table
   - Verify n values match expectations
   - Test with varying wavelengths

2. **Performance Profiling:**
   - Compare MPI vs ADK execution time
   - Check GPU kernel efficiency

3. **Documentation:**
   - Add example input files
   - Document C prefactor determination
   - Add to online documentation

4. **Future Enhancements:**
   - Add more species from literature
   - Support wavelength as runtime function (time-varying)
   - Implement cross-section based MPI rates (alternate to (C*E)^n)
   - Add ionization energy removal from fields (energy conservation)

---

## Timeline Estimate

- **Phase 1:** 2-4 hours (mostly careful copying/adapting initialization code)
- **Phase 2:** 3-5 hours (operator implementation, integration testing)
- **Total:** 5-9 hours for experienced WarpX developer

---

## Questions for Clarification

Before starting implementation, confirm:

1. **Units for prefactor C:** Should user provide in SI, or do we handle conversion?
2. **Default C value:** Is C=1.0 reasonable, or should it be species-dependent by default?
3. **Multiple ionization per step:** Should we allow n>1 ionizations per timestep, or enforce one at a time like ADK?
4. **Wavelength input:** Always wavelength in nm, or also allow frequency/photon energy input?

---

## References

- Existing ADK implementation: `Source/Particles/ElementaryProcess/Ionization.{H,cpp}`
- Field ionization integration: `FIELD_IONIZATION_INTEGRATION.md`
- AMReX filter-copy-transform: `Source/Particles/ParticleCreation/FilterCopyTransform.H`
- NIST Ionization Energies: `Source/Utils/Physics/IonizationEnergiesTable.H`

---

**End of Implementation Plan**

