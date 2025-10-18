# Custom Ionization Potentials Feature

## Overview
This feature allows users to override the default ionization potentials for any ionizable species in WarpX by specifying custom values in the input file.

## Implementation Details

### Modified Files
- `Source/Particles/PhysicalParticleContainer.cpp` (lines 1508-1532)

### Code Changes
Added support for the `ionization_potentials` parameter in the `InitIonizationModule()` function. The implementation:

1. Reads space-separated ionization potential values from the input file
2. Validates that the number of custom values doesn't exceed the atomic number
3. Overrides the default table values with custom ones
4. Prints the custom values to the output for verification

### Usage

In your input file, add the `ionization_potentials` parameter to any species with field ionization enabled:

```
species_name.do_field_ionization = 1
species_name.physical_element = C
species_name.ionization_potentials = 11.26 25.0 47.89 64.49 392.09 489.99
```

### Example: Carbon Ionization

For carbon (C, Z=6), the default ionization potentials are approximately:
- C+ (1st): 11.26 eV
- C2+ (2nd): 24.38 eV
- C3+ (3rd): 47.89 eV
- C4+ (4th): 64.49 eV
- C5+ (5th): 392.09 eV
- C6+ (6th): 489.99 eV

To override, for example, the first two levels:
```
carbon.do_field_ionization = 1
carbon.physical_element = C
carbon.ionization_potentials = 11.26 25.0
```

This will use 11.26 eV and 25.0 eV for the first two ionization levels, and keep the default values for higher levels.

### Test File
A complete test input file has been created at:
`test_custom_ionization_potentials.txt`

This demonstrates the feature with a carbon ionization example.

### Features
- **Space-separated values**: Easy to specify multiple potentials in one line
- **Partial override**: You can specify fewer potentials than the atomic number (unspecified levels use defaults)
- **Validation**: Checks that you don't specify more potentials than the atomic number
- **Output verification**: Prints custom values during initialization for easy verification
- **Works with all ionization models**: ADK, MPI (multiphoton), and PPT-ADK

### Error Handling
If you specify more ionization potentials than the atomic number, you'll get a clear error message:
```
Number of custom ionization potentials (N) cannot exceed atomic number (Z) for species 'species_name'
```

## Compilation
The code has been modified and is ready for compilation. To rebuild WarpX:
```bash
cd build
make -j4
```

Note: There may be unrelated compilation issues in the current build that need to be resolved separately (specifically in ParticleBoundaryBuffer.cpp).

## Next Steps
1. Fix any unrelated compilation issues in the build
2. Test with the provided test input file
3. Verify that custom ionization potentials are correctly applied during simulation
4. Consider adding this feature to the WarpX documentation
