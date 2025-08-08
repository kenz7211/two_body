"""
Orbital Models of High Velocity Stars in Omega Centauri
Using Octofitter
"""

# Environment variables
ENV["JULIA_NUM_THREADS"] = "auto"
ENV["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"


        
using Octofitter
using Distributions
using Unitful
using UnitfulAstro
using LinearAlgebra
using Octofitter  # Assuming a Julia equivalent or replace accordingly
using Statistics

# Add the directory to LOAD_PATH (good practice)
push!(LOAD_PATH, raw"C:\Users\macke\OneDrive - Saint Marys University\Summer Research 2025\two_body\orbit_utilities")
# Now you can use it
using octo_utils_Julia  # note the dot: it tells Julia this is a local module

# === 1. Select stars and time config ===
star_names = ["A", "B", "C", "D", "E", "F", "G"]
epoch = 2010.0
dt = 1.0

# Dictionaries to store simulation results and likelihood objects
epochs_mjd = Dict{String, Any}()
ra_rel = Dict{String, Any}()
dec_rel = Dict{String, Any}()
ra_errs = Dict{String, Any}()
dec_errs = Dict{String, Any}()
astrom_likelihoods = Dict{String, Any}()

# === 2. Simulate astrometry and create likelihood objects ===
for name in star_names
    star = octo_utils_Julia.stars[name]

    emjd, ra_r, dec_r, ra_e, dec_e = octo_utils_Julia.simulate_astrometry(star, epoch, dt)

    epochs_mjd[name] = emjd
    ra_rel[name] = ra_r
    dec_rel[name] = dec_r
    ra_errs[name] = ra_e
    dec_errs[name] = dec_e

    # Build tuple of observations, one per epoch
    obs = ntuple(i -> (
        epoch = emjd[i],
        ra = ra_r[i],
        dec = dec_r[i],
        σ_ra = ra_e[i],
        σ_dec = dec_e[i],
        cor = 0.0
    ), length(emjd))

    astrom_likelihoods[name] = PlanetRelAstromLikelihood(obs)
end

# Access via astrom_likelihoods["A"]
print(astrom_likelihoods["A"])
