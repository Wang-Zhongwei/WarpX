# Field Ionization Integration in WarpX

This document provides a comprehensive overview of how field ionization is integrated into the WarpX PIC code, from initialization through execution in the time-stepping loop.

## High-Level Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ INITIALIZATION                                                  │
│ WarpX::InitData()                                               │
│  └─ MultiParticleContainer::InitMultiPhysicsModules()           │
│      └─ PhysicalParticleContainer::InitIonizationModule()       │
│          • Read ionization energies table                       │
│          • Add "ionizationLevel" particle attribute             │
│          • Precompute ADK coefficients (with dt)                │
│          • Copy to GPU                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TIME STEPPING LOOP (each step)                                  │
│                                                                 │
│ WarpX::Evolve()                                                 │
│  ├─ doFieldIonization()  ← Uses E_aux, B_aux from prev step    │
│  │   └─ MultiParticleContainer::doFieldIonization()            │
│  │       └─ For each ionizable species:                        │
│  │           └─ For each particle tile:                        │
│  │               • Create IonizationFilterFunc (E, B, ADK)     │
│  │               • filterCopyTransformParticles:               │
│  │                   ├─ Filter: compute P, random draw         │
│  │                   ├─ Copy: create electron if ionized       │
│  │                   └─ Transform: increment ionizationLevel   │
│  ├─ doQEDEvents()                                               │
│  └─ OneStep()                                                   │
│      ├─ doCollisions()                                          │
│      ├─ PushParticlesandDeposit()                              │
│      │   • Charge/current scaled by ionizationLevel            │
│      └─ Field solve (Maxwell)                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Initialization Phase

The initialization phase sets up the ionization module for each species that requires field ionization. This happens during the startup sequence before the main time-stepping loop begins. The key tasks include reading atomic data tables, precomputing ADK coefficients, and allocating particle attributes to track ionization states.

### 1.1 Configuration (Input Files or PICMI)

Users can enable field ionization either through input files or the PICMI Python interface. The configuration specifies which chemical element is being ionized, the initial ionization state, and which species will receive the liberated electrons.

**PICMI Interface**: `Python/pywarpx/picmi.py:2503-2520`
```python
class FieldIonization(picmistandard.PICMI_FieldIonization):
    """
    WarpX only has ADK ionization model implemented.
    """
    def interaction_initialize_inputs(self):
        assert self.model == "ADK", "WarpX only has ADK ionization model implemented"
        self.ionized_species.species.do_field_ionization = 1
        self.ionized_species.species.physical_element = (
            self.ionized_species.particle_type
        )
        self.ionized_species.species.ionization_product_species = (
            self.product_species.name
        )
        self.ionized_species.species.ionization_initial_level = (
            self.ionized_species.charge_state
        )
        self.ionized_species.species.charge = "q_e"
```

### 1.2 Module Initialization Sequence

The initialization follows a hierarchical call pattern from the top-level WarpX initialization down to individual particle containers. The critical requirement is that the timestep `dt` must be computed before ionization initialization because the ADK rate coefficients depend on `dt`.

**Main Entry Point**: `Source/Initialization/WarpXInitData.cpp:732`
```cpp
void WarpX::InitData ()
{
    // ... other initialization ...
    if (restart_chkfile.empty())
    {
        ComputeDt();
        ::PrintDtDxDyDz(max_level, geom, dt);
        InitFromScratch();
        InitDiagnostics();
    }
    // ... rest of initialization ...
}
```

**Particle Container Initialization**: `Source/Particles/MultiParticleContainer.cpp:416-422`
```cpp
void MultiParticleContainer::InitData ()
{
    InitMultiPhysicsModules();

    for (auto& pc : allcontainers) {
        pc->InitData();
    }
}
```

**Multi-Physics Modules Setup**: `Source/Particles/MultiParticleContainer.cpp:436-451`
```cpp
void MultiParticleContainer::InitMultiPhysicsModules ()
{
    // Init ionization module here instead of in the MultiParticleContainer
    // constructor because dt is required to compute ionization rate pre-factors
    for (auto& pc : allcontainers) {
        pc->InitIonizationModule();
    }
    // For each species, get the ID of its product species.
    // This is used for ionization and pair creation processes.
    mapSpeciesProduct();
    CheckIonizationProductSpecies();
#ifdef WARPX_QED
    CheckQEDProductSpecies();
    InitQED();
#endif
}
```

### 1.3 Per-Species Ionization Module Initialization

This is the core initialization routine where the heavy computation occurs. For each ionizable species, this function:
1. Validates that the species charge is set to elementary charge `q_e`
2. Reads the element's atomic number and ionization energies from precompiled NIST tables
3. Adds an integer particle attribute `ionizationLevel` to track each particle's current ionization state
4. Computes ADK prefactors, powers, and exponential coefficients for each ionization level (0 to Z)
5. Transfers all tables to GPU memory for fast access during simulation

The ADK coefficients are precomputed here (rather than on-the-fly) to minimize computational cost during the time-stepping loop.

**Core Initialization**: `Source/Particles/PhysicalParticleContainer.cpp:1445-1527`
```cpp
void PhysicalParticleContainer::InitIonizationModule ()
{
    if (!do_field_ionization) { return; }
    const ParmParse pp_species_name(species_name);
    if (charge != PhysConst::q_e){
        ablastr::warn_manager::WMRecordWarning("Species",
            "charge != q_e for ionizable species '" +
            species_name + "':" +
            "overriding user value and setting charge = q_e.");
        charge = PhysConst::q_e;
    }
    utils::parser::queryWithParser(pp_species_name, "do_adk_correction", do_adk_correction);

    utils::parser::queryWithParser(
        pp_species_name, "ionization_initial_level", ionization_initial_level);
    pp_species_name.get("ionization_product_species", ionization_product_name);
    pp_species_name.get("physical_element", physical_element);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        physical_element == "H" || !do_adk_correction,
        "Correction to ADK by Zhang et al., PRA 90, 043410 (2014) only works with Hydrogen");
    // Add runtime integer component for ionization level
    AddIntComp("ionizationLevel");
    // Get atomic number and ionization energies from file
    const int ion_element_id = utils::physics::ion_map_ids.at(physical_element);
    ion_atomic_number = utils::physics::ion_atomic_numbers[ion_element_id];
    Vector<Real> h_ionization_energies(ion_atomic_number);
    const int offset = utils::physics::ion_energy_offsets[ion_element_id];
    for(int i=0; i<ion_atomic_number; i++){
        h_ionization_energies[i] =
            utils::physics::table_ionization_energies[i+offset];
    }
    // Compute ADK prefactors (See Chen, JCP 236 (2013), equation (2))
    // For now, we assume l=0 and m=0.
    // The approximate expressions are used,
    // without Gamma function
    constexpr auto a3 = PhysConst::alpha*PhysConst::alpha*PhysConst::alpha;
    constexpr auto a4 = a3 * PhysConst::alpha;
    constexpr Real wa = a3 * PhysConst::c / PhysConst::r_e;
    constexpr Real Ea = PhysConst::m_e * PhysConst::c*PhysConst::c /PhysConst::q_e *
        a4/PhysConst::r_e;
    constexpr Real UH = utils::physics::table_ionization_energies[0];
    const Real l_eff = std::sqrt(UH/h_ionization_energies[0]) - 1._rt;

    const Real dt = WarpX::GetInstance().getdt(0);

    ionization_energies.resize(ion_atomic_number);
    adk_power.resize(ion_atomic_number);
    adk_prefactor.resize(ion_atomic_number);
    adk_exp_prefactor.resize(ion_atomic_number);

    Gpu::copyAsync(Gpu::hostToDevice,
                   h_ionization_energies.begin(), h_ionization_energies.end(),
                   ionization_energies.begin());

    adk_correction_factors.resize(4);
    if (do_adk_correction) {
        Vector<Real> h_correction_factors(4);
        constexpr int offset_corr = 0; // hard-coded: only Hydrogen
        for(int i=0; i<4; i++){
            h_correction_factors[i] = table_correction_factors[i+offset_corr];
        }
        Gpu::copyAsync(Gpu::hostToDevice,
                       h_correction_factors.begin(), h_correction_factors.end(),
                       adk_correction_factors.begin());
    }

    Real const* AMREX_RESTRICT p_ionization_energies = ionization_energies.data();
    Real * AMREX_RESTRICT p_adk_power = adk_power.data();
    Real * AMREX_RESTRICT p_adk_prefactor = adk_prefactor.data();
    Real * AMREX_RESTRICT p_adk_exp_prefactor = adk_exp_prefactor.data();
    amrex::ParallelFor(ion_atomic_number, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        const Real n_eff = (i+1) * std::sqrt(UH/p_ionization_energies[i]);
        const Real C2 = std::pow(2._rt,2._rt*n_eff)/(n_eff*std::tgamma(n_eff+l_eff+1._rt)*std::tgamma(n_eff-l_eff));
        p_adk_power[i] = -(2._rt*n_eff - 1._rt);
        const Real Uion = p_ionization_energies[i];
        p_adk_prefactor[i] = dt * wa * C2 * ( Uion/(2._rt*UH) )
            * std::pow(2._rt*std::pow((Uion/UH),3._rt/2._rt)*Ea,2._rt*n_eff - 1._rt);
        p_adk_exp_prefactor[i] = -2._rt/3._rt * std::pow( Uion/UH,3._rt/2._rt) * Ea;
    });

    Gpu::synchronize();
}
```

**Ionization Energies Table**: `Source/Utils/Physics/IonizationEnergiesTable.H:1-1973`
- Contains atomic data for all elements (H through U)
- Generated from NIST data via `Source/Utils/Physics/write_atomic_data_cpp.py`
- Provides ionization potentials for all charge states of each element

## 2. Execution Phase (Time Stepping Loop)

During each timestep, field ionization is evaluated before the particle push and field solve. This ordering is important because:
1. Ionization depends on the electromagnetic fields from the previous timestep (stored in `_aux` arrays)
2. Newly created electrons need to be included in the current deposition and field solve
3. The ionization state affects the effective charge used in deposition

The execution flow processes all ionizable species, evaluates the ADK ionization probability for each particle based on local field strength, and creates electrons stochastically according to the computed probability.

### 2.1 Main Evolution Loop

At the top level, `doFieldIonization()` is called early in each timestep, before collisions and the main PIC push-deposit-solve cycle. This ensures that ionization events from the current timestep's fields are accounted for in the subsequent particle and field updates.

**Time Step Entry Point**: `Source/Evolve/WarpXEvolve.cpp:219-229`
```cpp
// multi-physics: field ionization
doFieldIonization();

#ifdef WARPX_QED
// multi-physics: QED effects
doQEDEvents();
mypc->doQEDSchwinger();
#endif

// perform collisions and advance fields and particles by one time step
OneStep(cur_time, dt[0], step);
```

### 2.2 Field Ionization Execution

The `doFieldIonization()` method operates on all refinement levels, passing the auxiliary field arrays (`_aux`) which contain the electromagnetic fields from the previous timestep. These fields are used rather than the current fields because they are guaranteed to be synchronized and available at this point in the algorithm.

**WarpX Level Call**: `Source/Evolve/WarpXEvolve.cpp:1172-1188`
```cpp
void WarpX::doFieldIonization ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    for (int lev = 0; lev <= finest_level; ++lev) {
        mypc->doFieldIonization(
            lev,
            *m_fields.get(FieldType::Efield_aux, Direction{0}, lev),
            *m_fields.get(FieldType::Efield_aux, Direction{1}, lev),
            *m_fields.get(FieldType::Efield_aux, Direction{2}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{0}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{1}, lev),
            *m_fields.get(FieldType::Bfield_aux, Direction{2}, lev)
        );
    }
}
```

**MultiParticleContainer Level**: `Source/Particles/MultiParticleContainer.cpp:1001-1055`

This function orchestrates ionization across all species. For each ionizable species, it:
1. Identifies the product species (typically electrons) where liberated particles will be added
2. Creates filter, copy, and transform functors that define the ionization logic
3. Iterates over particle tiles in parallel (using OpenMP/GPU parallelization)
4. For each tile, applies the filter-copy-transform pattern to evaluate and execute ionization events
5. Assigns unique particle IDs to newly created electrons
```cpp
void MultiParticleContainer::doFieldIonization (int lev,
                                               const MultiFab& Ex,
                                               const MultiFab& Ey,
                                               const MultiFab& Ez,
                                               const MultiFab& Bx,
                                               const MultiFab& By,
                                               const MultiFab& Bz)
{
    WARPX_PROFILE("MultiParticleContainer::doFieldIonization()");

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Loop over all species.
    // Ionized particles in pc_source create particles in pc_product
    for (auto& pc_source : allcontainers)
    {
        if (!pc_source->do_field_ionization){ continue; }

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

            auto Filter = phys_pc_ptr->getIonizationFunc(pti, lev, Ex.nGrowVect(),
                                                         Ex[pti], Ey[pti], Ez[pti],
                                                         Bx[pti], By[pti], Bz[pti]);

            const auto np_dst = dst_tile.numParticles();
            const auto num_added = filterCopyTransformParticles<1>(*pc_product, dst_tile, src_tile, np_dst,
                                                                   Filter, Copy, Transform);

            setNewParticleIDs(dst_tile, np_dst, num_added);
```

### 2.3 Ionization Filter Function Creation

The ionization filter function is constructed per particle tile and captures all data needed to evaluate ADK ionization probabilities on the GPU. This includes references to the field arrays, precomputed ADK tables, and grid geometry information. The filter function will be called for each particle to determine whether it undergoes ionization.

**Filter Function Factory**: `Source/Particles/PhysicalParticleContainer.cpp:1529-1552`
```cpp
IonizationFilterFunc
PhysicalParticleContainer::getIonizationFunc (const WarpXParIter& pti,
                                              int lev,
                                              amrex::IntVect ngEB,
                                              const amrex::FArrayBox& Ex,
                                              const amrex::FArrayBox& Ey,
                                              const amrex::FArrayBox& Ez,
                                              const amrex::FArrayBox& Bx,
                                              const amrex::FArrayBox& By,
                                              const amrex::FArrayBox& Bz)
{
    WARPX_PROFILE("PhysicalParticleContainer::getIonizationFunc()");

    return {pti, lev, ngEB, Ex, Ey, Ez, Bx, By, Bz,
                                m_E_external_particle, m_B_external_particle,
                                ionization_energies.dataPtr(),
                                adk_prefactor.dataPtr(),
                                adk_exp_prefactor.dataPtr(),
                                adk_power.dataPtr(),
                                adk_correction_factors.dataPtr(),
                                GetIntCompIndex("ionizationLevel"),
                                ion_atomic_number,
                                do_adk_correction};
}
```

**Filter Function Constructor**: `Source/Particles/ElementaryProcess/Ionization.cpp:20-85`

The constructor stores pointers to all required data and sets up field interpolation machinery. It captures:
- Field array pointers and their index types (for proper staggering)
- ADK coefficient arrays for all ionization levels
- External field contributions
- Grid spacing and lower corner coordinates for particle-to-grid mapping
- Interpolation order and Galerkin settings
```cpp
IonizationFilterFunc::IonizationFilterFunc (const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
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
                                            int a_comp,
                                            int a_atomic_number,
                                            int a_do_adk_correction,
                                            int a_offset) noexcept:
    m_ionization_energies{a_ionization_energies},
    m_adk_prefactor{a_adk_prefactor},
    m_adk_exp_prefactor{a_adk_exp_prefactor},
    m_adk_power{a_adk_power},
    m_adk_correction_factors{a_adk_correction_factors},
    comp{a_comp},
    m_atomic_number{a_atomic_number},
    m_do_adk_correction{a_do_adk_correction},
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

    m_get_position  = GetParticlePosition<PIdx>(a_pti, a_offset);
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

    // Lower corner of tile box physical domain (take into account Galilean shift)
    m_xyzmin = WarpX::LowerCorner(box, lev, 0._rt);

    m_lo = amrex::lbound(box);
}
```

### 2.4 ADK Ionization Probability Calculation

This is the heart of the field ionization physics. For each ion particle, the filter operator:
1. **Checks ionization level**: Returns false if already fully ionized
2. **Gathers E and B fields**: Uses shape function interpolation to get fields at particle position
3. **Lorentz transforms to rest frame**: Computes the electric field magnitude in the particle's instantaneous rest frame using the relativistic field transformation (critical for boosted frame simulations and relativistic particles)
4. **Evaluates ADK formula**: Computes the ionization rate W using the Ammosov-Delone-Krainov tunneling ionization model
5. **Applies Zhang correction** (optional, H only): Empirical extension for over-the-barrier regime
6. **Converts to probability**: Transforms rate to probability over timestep: P = 1 - exp(-W·dt/γ)
7. **Stochastic decision**: Draws random number and returns true if ionization occurs

**Core ADK Implementation**: `Source/Particles/ElementaryProcess/Ionization.H:75-158`
```cpp
template <typename PData>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool operator() (const PData& ptd, int i, amrex::RandomEngine const& engine) const noexcept
{
    using namespace amrex::literals;
    using namespace WarpXConst;

    const int ion_lev = ptd.m_runtime_idata[comp][i];
    if (ion_lev >= m_atomic_number) { return false; }

    amrex::ParticleReal xp, yp, zp;
    m_get_position(i, xp, yp, zp);

    amrex::ParticleReal ex, ey, ez, bx, by, bz;
    m_get_externalEB(i, ex, ey, ez, bx, by, bz);

    // gather E and B
    doGatherShapeN<1>(xp, yp, zp, ex, ey, ez, bx, by, bz,
                      m_ex_arr, m_ey_arr, m_ez_arr, m_bx_arr, m_by_arr, m_bz_arr,
                      m_ex_type, m_ey_type, m_ez_type, m_bx_type, m_by_type, m_bz_type,
                      m_dinv, m_xyzmin, m_lo, m_n_rz_azimuthal_modes,
                      m_nox, m_galerkin_interpolation);

    // Compute electric field amplitude in the particle's frame of
    // reference (particularly important when in boosted frame).
    // This is the LORENTZ TRANSFORMATION of electromagnetic fields to the particle's rest frame.
    const amrex::ParticleReal ux = ptd.m_rdata[PIdx::ux][i];
    const amrex::ParticleReal uy = ptd.m_rdata[PIdx::uy][i];
    const amrex::ParticleReal uz = ptd.m_rdata[PIdx::uz][i];

    const auto ga = static_cast<amrex::Real>(
        std::sqrt(1. + (ux*ux + uy*uy + uz*uz) * c2_inv));  // Lorentz factor γ
    
    // Electric field in particle rest frame using relativistic field transformation:
    // E'² = -1/c² (u·E)² + (γE + u×B)²
    // This accounts for both Lorentz contraction and the velocity-dependent transformation
    const amrex::Real E = std::sqrt(
                           - ( ux*ex + uy*ey + uz*ez ) * ( ux*ex + uy*ey + uz*ez ) * c2_inv
                           + ( ga   *ex + uy*bz - uz*by ) * ( ga   *ex + uy*bz - uz*by )
                           + ( ga   *ey + uz*bx - ux*bz ) * ( ga   *ey + uz*bx - ux*bz )
                           + ( ga   *ez + ux*by - uy*bx ) * ( ga   *ez + ux*by - uy*bx )
                           );

    // Compute probability of ionization p
    amrex::Real w_dtau = (E <= 0._rt) ? 0._rt : 1._rt/ ga * m_adk_prefactor[ion_lev] *
        std::pow(E, m_adk_power[ion_lev]) *
        std::exp( m_adk_exp_prefactor[ion_lev]/E );
    // if requested, do Zhang's correction of ADK
    if (m_do_adk_correction) {
        const amrex::Real r = E / m_adk_correction_factors[3];
        w_dtau *= std::exp(m_adk_correction_factors[0]*r*r+m_adk_correction_factors[1]*r+
                           m_adk_correction_factors[2]);
    }

    const amrex::Real p = 1._rt - std::exp( - w_dtau );

    const amrex::Real random_draw = amrex::Random(engine);
    if (random_draw < p)
    {
        return true;
    }
    return false;
}
```

### 2.5 Particle Creation and Transformation

The filter-copy-transform pattern is a general-purpose GPU-parallelizable framework for conditional particle creation. It efficiently handles:
1. **Filtering**: Applies the ionization probability test to generate a mask (true/false for each particle)
2. **Parallel scan**: Computes offsets for compacting particles (only store those that passed the filter)
3. **Copying**: Creates new electrons with same position and momentum as the parent ion
4. **Transforming**: Updates the parent ion's ionization level by incrementing its `ionizationLevel` attribute

This pattern avoids atomic operations and branch divergence, making it well-suited for GPU execution.

**Filter-Copy-Transform Pattern**: `Source/Particles/ParticleCreation/FilterCopyTransform.H:56-106`
```cpp
template <int N, typename DstPC, typename DstTile, typename SrcTile, typename Index,
          typename TransFunc, typename CopyFunc,
          amrex::EnableIf_t<std::is_integral_v<Index>, int> foo = 0>
Index filterCopyTransformParticles (DstPC& pc, DstTile& dst, SrcTile& src,
                                    Index* mask, Index dst_index,
                                    CopyFunc&& copy, TransFunc&& transform) noexcept
{
    using namespace amrex;

    const auto np = src.numParticles();
    if (np == 0) { return 0; }

    Gpu::DeviceVector<Index> offsets(np);
    auto total = amrex::Scan::ExclusiveSum(np, mask, offsets.data());
    const Index num_added = N * total;
    auto old_np = dst.size();
    auto new_np = std::max(dst_index + num_added, dst.numParticles());
    dst.resize(new_np);

    auto *const p_offsets = offsets.dataPtr();

    const auto src_data = src.getParticleTileData();
    const auto dst_data = dst.getParticleTileData();

    amrex::ParallelForRNG(np,
    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
    {
        if (mask[i])
        {
            for (int j = 0; j < N; ++j) {
                copy(dst_data, src_data, i, N*p_offsets[i] + dst_index + j, engine);
            }
            transform(dst_data, src_data, i, N*p_offsets[i] + dst_index, engine);
        }
    });

    ParticleCreation::DefaultInitializeRuntimeAttributes(dst,
                                       0, 0,
                                       pc.getUserRealAttribs(), pc.getUserIntAttribs(),
                                       pc.GetRealSoANames(), pc.GetIntSoANames(),
                                       pc.getUserRealAttribParser(),
                                       pc.getUserIntAttribParser(),
#ifdef WARPX_QED
                                       false,
                                              // when calling the CopyFunc functor
                                       pc.get_breit_wheeler_engine_ptr(),
                                       pc.get_quantum_sync_engine_ptr(),
#endif
                                       pc.getIonizationInitialLevel(),
                                       old_np, new_np);

    Gpu::synchronize();
    return num_added;
}
```

**Ionization Transform Function**: `Source/Particles/ElementaryProcess/Ionization.H:161-171`

This simple functor increments the ionization level of the parent ion particle. Each time a particle ionizes, its `ionizationLevel` increases by 1, representing one additional electron removed. The ionization level is then used during deposition to scale the effective charge (q_eff = q_e × ionizationLevel).
```cpp
struct IonizationTransformFunc
{
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& /*dst*/, SrcData& src,
        int i_src, int /*i_dst*/,
        amrex::RandomEngine const& /*engine*/) const noexcept
    {
        src.m_runtime_idata[0][i_src] += 1;
    }
};
```

## 3. Integration with Other Physics

The ionization module integrates seamlessly with other parts of the PIC cycle by modifying how particles contribute to charge and current densities based on their ionization state.

### 3.1 Charge Deposition

When depositing charge to the grid, ions use their ionization level to compute the effective charge. A nitrogen ion at ionization level 3 (N³⁺) will deposit charge as if it were 3 elementary charges. This ensures that the total charge conservation is maintained as electrons are liberated from parent ions.

**Ionization Level Scaling**: `Source/Particles/Deposition/ChargeDeposition.H:50-71`
```cpp
// Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
// (do_ionization=1)
const bool do_ionization = ion_lev;

const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

amrex::Array4<amrex::Real> const& rho_arr = rho_fab.array();
amrex::IntVect const rho_type = rho_fab.box().type();

// Loop over particles and deposit into rho_fab
amrex::ParallelFor(
        np_to_deposit,
        [=] AMREX_GPU_DEVICE (long ip) {
        // --- Get particle quantities
        amrex::Real wq = q*wp[ip]*invvol;
        if (do_ionization){
            wq *= ion_lev[ip];
        }
```

### 3.2 Current Deposition

Similar scaling applies to current deposition in `Source/Particles/Deposition/CurrentDeposition.H` (lines 311-379). The current density j = qnv is scaled by the ionization level to ensure proper electromagnetic field evolution.

### 3.3 Particle Creation

When particles are created through injection or other mechanisms (not ionization), they are initialized with the species' `ionization_initial_level`. This allows starting simulations with partially pre-ionized plasmas.

**Default Initialization**: `Source/Particles/ParticleCreation/DefaultInitialization.H:245-258`
```cpp
// Current runtime comp is ionization level
auto const it_ioniz = std::find(particle_icomps.begin(), particle_icomps.end(), "ionizationLevel");
if (it_ioniz != particle_icomps.end() &&
    std::distance(particle_icomps.begin(), it_ioniz) == j)
{
    if (soa.GetIntData(particle_icomps[j]).size() > 0) {
        auto attr_ptr = soa.GetIntData(particle_icomps[j]).data() + old_size;
        for (int ip = 0; ip < num_added; ++ip) {
            attr_ptr[ip] = ionization_initial_level;
        }
    } else {
        for (int ip = 0; ip < num_added; ++ip) {
            attr_ptr[ip] = ionization_initial_level;
        }
    }
}
```

## 4. Data Structures

Understanding the data structures is key to following the code. Ionization data is stored at multiple levels: per-species containers hold atomic data and ADK tables, while individual particles carry their current ionization state.

### 4.1 Per-Species Container Data

These member variables are stored in the particle container for each ionizable species:

**WarpXParticleContainer Members**: `Source/Particles/WarpXParticleContainer.H:626-631`
```cpp
int do_field_ionization = 0;
int ionization_product;
std::string ionization_product_name;
int ionization_initial_level = 0;
amrex::Gpu::DeviceVector<amrex::Real> ionization_energies;
```

**PhysicalParticleContainer Members**: `Source/Particles/PhysicalParticleContainer.H:186`
```cpp
IonizationFilterFunc getIonizationFunc (const WarpXParIter& pti,
                                        int lev,
                                        amrex::IntVect ngEB,
                                        const amrex::FArrayBox& Ex,
                                        const amrex::FArrayBox& Ey,
                                        const amrex::FArrayBox& Ez,
                                        const amrex::FArrayBox& Bx,
                                        const amrex::FArrayBox& By,
                                        const amrex::FArrayBox& Bz);
```

### 4.2 Per-Particle Attributes

Each ion particle carries an integer attribute:
- `ionizationLevel` (integer) - Current ionization state ranging from 0 (neutral) to Z (fully stripped)

This attribute is dynamically updated each timestep as ionization events occur and is used during deposition to compute effective charge.

## 5. Physics Model

The physics implementation follows the Ammosov-Delone-Krainov (ADK) model for tunnel ionization, with an optional empirical correction for hydrogen in the over-the-barrier regime.

### 5.1 ADK Theory Implementation

The ADK theory requires evaluating the electric field in the particle's rest frame, which is particularly important for:
- Boosted-frame simulations (common in laser-plasma acceleration)
- Relativistic ions
- Accurate modeling of ionization in strong crossed fields

**Electric field in particle frame (Lorentz Transformation)**: `Source/Particles/ElementaryProcess/Ionization.H:131-136`
```cpp
const amrex::Real E = std::sqrt(
                   - ( ux*ex + uy*ey + uz*ez ) * ( ux*ex + uy*ey + uz*ez ) * c2_inv
                   + ( ga   *ex + uy*bz - uz*by ) * ( ga   *ex + uy*bz - uz*by )
                   + ( ga   *ey + uz*bx - ux*bz ) * ( ga   *ey + uz*bx - ux*bz )
                   + ( ga   *ez + ux*by - uy*bx ) * ( ga   *ez + ux*by - uy*bx )
                   );
```

**Ionization rate calculation (ADK formula)**: `Source/Particles/ElementaryProcess/Ionization.H:139-147`

The ionization rate W is computed using the ADK formula with precomputed coefficients. The rate depends on:
- Electric field strength E (in particle frame)
- Current ionization level (determines which ADK coefficients to use)
- Ionization potential of the current state
- Optional Zhang correction for hydrogen
```cpp
// Compute probability of ionization p
amrex::Real w_dtau = (E <= 0._rt) ? 0._rt : 1._rt/ ga * m_adk_prefactor[ion_lev] *
    std::pow(E, m_adk_power[ion_lev]) *
    std::exp( m_adk_exp_prefactor[ion_lev]/E );
// if requested, do Zhang's correction of ADK
if (m_do_adk_correction) {
    const amrex::Real r = E / m_adk_correction_factors[3];
    w_dtau *= std::exp(m_adk_correction_factors[0]*r*r+m_adk_correction_factors[1]*r+
                       m_adk_correction_factors[2]);
}
```

**Probability over timestep (stochastic sampling)**: `Source/Particles/ElementaryProcess/Ionization.H:149-152`

The ionization rate W (probability per unit time) is converted to a probability P over the finite timestep dt. The factor of 1/γ accounts for time dilation in the particle's frame. A uniform random number is drawn and compared to P to decide whether ionization occurs for this particle this timestep.
```cpp
const amrex::Real p = 1._rt - std::exp( - w_dtau );

const amrex::Real random_draw = amrex::Random(engine);
if (random_draw < p)
{
    return true;
}
```

### 5.2 Theory Documentation

**Complete ADK equations**: `Docs/source/theory/multiphysics/ionization.rst:30-65`

The theory documentation provides the full mathematical derivation of the ADK formula, including:
- Definition of effective quantum numbers n* and l*
- Atomic unit conversions
- Barrier suppression field calculations
- References to original papers (Ammosov 1986, Chen 2013, Zhang 2014)

## 6. Key Assumptions

These assumptions are built into the current implementation:

1. **Angular momentum quantum numbers**: l = 0, m = 0
2. **One ionization level per timestep**: Enforced by filter returning true once
3. **Energy conservation**: Ionization energy not removed from EM fields
4. **Field timing**: Uses E and B fields from previous timestep (stored in `_aux` arrays)
5. **Element support**: All elements H through U via NIST data tables

## 7. Performance Considerations

The implementation is designed for performance on modern heterogeneous architectures:

- **GPU parallelization**: All particle operations use AMReX GPU kernels with `AMREX_GPU_DEVICE` qualifiers
- **Precomputed tables**: ADK coefficients calculated once during initialization (O(Z) operations) rather than every timestep (O(N_particles) operations)
- **Memory efficiency**: Ionization level stored as single integer per particle (4 bytes overhead per ion)
- **Coalesced memory access**: Structure-of-arrays layout ensures efficient GPU memory transactions
- **Load balancing**: Ionization costs tracked via `WARPX_PROFILE` and used for dynamic load balancing across MPI ranks
- **Minimal branching**: Filter-copy-transform pattern uses mask-based selection to minimize warp divergence

## 8. Validation and Testing

Field ionization functionality is tested through:
- **Regression tests**: `Examples/Tests/field_ionization/` with reference checksums
- **2D simulations sufficient**: Full physics can be validated without 3D computational expense
- **Comparison with theory**: ADK rates verified against analytical predictions
- **Charge conservation**: Total charge monitored throughout simulation

## 9. Summary

Field ionization in WarpX is implemented as a tightly integrated multi-physics module that:
1. **Initializes** during startup by reading NIST atomic data and precomputing ADK coefficients
2. **Executes** early in each timestep using fields from the previous step
3. **Creates** electrons stochastically based on ADK tunnel ionization theory
4. **Updates** ion charge states by incrementing the ionization level
5. **Integrates** with deposition routines through ionization-level-dependent effective charge

The implementation leverages GPU parallelization, precomputed tables, and efficient data structures to achieve high performance while maintaining physical accuracy through proper Lorentz transformations and quantum tunneling physics.
