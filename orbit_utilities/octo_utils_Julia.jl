module octo_utils_Julia

# ========== Environment variables ==========
ENV["JULIA_NUM_THREADS"] = "auto"
ENV["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"

# ========== Imports ==========

using Unitful
using UnitfulAstro
using LinearAlgebra
using Octofitter  # Assuming a Julia equivalent or replace accordingly
using Statistics


# ========== Define StarData Type ==========
"""
Utilities for organizing and fitting multi-star astrometric data with Octofitter.
Includes:
- StarData struct for storing motion and error data
- stars dictionary with input for each star
- Functions to build models, propagate errors, and run Octofitter fits and plots
"""
struct StarData
    name::String
    ra::Float64         # Right Ascension position in milliarcseconds (mas)
    dec::Float64        # Declination position in mas
    pm_ra::Float64      # Proper motion in RA (mas/year)
    pm_dec::Float64     # Proper motion in Dec (mas/year)
    acc_ra::Float64     # Acceleration in RA (mas/year²)
    acc_dec::Float64    # Acceleration in Dec (mas/year²)
    sigma_pm_ra::Float64    # Uncertainty in proper motion RA (mas/year)
    sigma_pm_dec::Float64   # Uncertainty in proper motion Dec (mas/year)
    sigma_acc_ra::Float64   # Uncertainty in acceleration RA (mas/year²)
    sigma_acc_dec::Float64  # Uncertainty in acceleration Dec (mas/year²)
    v2D::Float64            # 2D velocity (km/s)
    v2D_err::Float64        # Uncertainty in 2D velocity (km/s)
end

# ========== Omega Centauri Data ==========
# We find the center to be at (α,δ) = (13:26:47.24, −47:28:46.45).
# from https://iopscience.iop.org/article/10.1088/0004-637X/710/2/1032
# reference 6 in Haberle et al. (2024; Nature, Vol. 631) 
# We assume the IMBH is located at the cluster center

# Center of mass RA and Dec in mas
ra_cm_mas  = u"deg"(201.696834) |> x -> u"mas"(x) |> ustrip
dec_cm_mas = u"deg"(-47.479569) |> x -> u"mas"(x) |> ustrip

# Distance to Omega Centauri center (kiloparsecs and converted to km)
distance_kpc = 5.43u"kpc"
distance_km = uconvert(u"km", distance_kpc)

# Assumed errors in position (mas)
ra_err = 0.5u"mas"
dec_err = 0.5u"mas"

# ========== Define Error Propagation Function ==========
"""
Propagates uncertainty in position due to uncertainties in
proper motion and acceleration over time. No uncertainty in initial position is available.

Parameters:
- sigma_pos : Initial uncertainty in position (mas)
- sigma_pm  : Uncertainty in proper motion (mas/yr)
- sigma_acc : Uncertainty in acceleration (mas/yr²)
- dt        : Time from reference epoch (in years)

Returns:
- sigma_pos : Total uncertainty in predicted position at time dt (mas)
"""
function propagate_error(sigma_pos, sigma_pm, sigma_acc, dt)
    term_pos = sigma_pos^2                      # variance from initial position
    term_pm = (dt * sigma_pm)^2                 # variance from proper motion
    term_acc = (0.5 * dt^2 * sigma_acc)^2      # variance from acceleration

    sqrt(term_pos + term_pm + term_acc)
end

# ========== Define Position Projection Function ==========
"""
Gives a fake observed position using an observed angular position, velocity, and acceleration

Parameters:
- pos : Initial position (in mas)
- pm : Proper motion (in mas/yr)
- acc : Acceleration (in mas/yr²)
- dt : Time offset(s) from the reference epoch in years

Returns:
- pos_final : calculated position (in mas)
"""
function fake_pos(pos, pm, acc, dt)
    pos + pm*dt + 0.5*acc*dt^2
end

# ========== Define Star Dictionary ==========
stars = Dict{String,StarData}(
    "A" => StarData(
        "A",
        u"deg"(201.6967263) |> x -> u"mas"(x) |> ustrip, # mas
        u"deg"(-47.4795835) |> x -> u"mas"(x) |> ustrip, # mas
        3.563, # mas/year
        2.564, # mas/year
        -0.0069, # mas/year²
        0.0085,  # mas/year²
        0.038,   # mas/year
        0.055,   # mas/year
        0.0083,  # mas/year²
        0.0098,  # mas/year²
        113.0,   # km/s
        1.1      # km/s error
    ),
    "B" => StarData(
        "B",
        u"deg"(201.6968888) |> x -> u"mas"(x) |> ustrip, # mas
        u"deg"(-47.4797138) |> x -> u"mas"(x) |> ustrip, # mas
        2.167,
        1.415,
        0.0702,
        0.0228,
        0.182,
        0.081,
        0.0239,
        0.0157,
        66.6,
        4.1
    ),
    "C" => StarData(
        "C",
        u"deg"(201.6966378) |> x -> u"mas"(x) |> ustrip,
        u"deg"(-47.4793672) |> x -> u"mas"(x) |> ustrip,
        1.117,
        3.514,
        0.0028,
        -0.0060,
        0.127,
        0.056,
        0.0333,
        0.0123,
        94.9,
        1.7
    ),
    "D" => StarData(
        "D",
        u"deg"(201.6968346) |> x -> u"mas"(x) |> ustrip,
        u"deg"(-47.4793233) |> x -> u"mas"(x) |> ustrip,
        2.559,
        -1.617,
        0.0357,
        -0.0194,
        0.082,
        0.061,
        0.0177,
        0.0162,
        77.9,
        2.0
    ),
    "E" => StarData(
        "E",
        u"deg"(201.6973080) |> x -> u"mas"(x) |> ustrip,
        u"deg"(-47.4797545) |> x -> u"mas"(x) |> ustrip,
        -2.149,
        1.638,
        0.0072,
        -0.0009,
        0.025,
        0.037,
        0.0042,
        0.0075,
        69.6,
        0.8
    ),
    "F" => StarData(
        "F",
        u"deg"(201.6977125) |> x -> u"mas"(x) |> ustrip,
        u"deg"(-47.4792625) |> x -> u"mas"(x) |> ustrip,
        0.436,
        -2.584,
        0.0052,
        -0.0015,
        0.017,
        0.016,
        0.0038,
        0.0038,
        67.4,
        0.4
    ),
    "G" => StarData(
        "G",
        u"deg"(201.6961340) |> x -> u"mas"(x) |> ustrip,
        u"deg"(-47.4790585) |> x -> u"mas"(x) |> ustrip,
        -1.317,
        2.207,
        -0.0197,
        0.0173,
        0.098,
        0.062,
        0.0267,
        0.0170,
        66.2,
        1.9
    ),
)

# ========== Astrometry Input for Octofitter ==========
"""
Simulates astrometric positions for a star at three epochs: past, present, and future.

Parameters:
- star: StarData object containing motion and error data
- epoch: Central epoch in calendar years (e.g., 2010)
- dt: Time offset from the central epoch in years (e.g., 10)

Returns:
- epochs_mjd: Modified Julian Dates for the three epochs
- ra_rel: Relative right ascension values at the three epochs (mas)
- dec_rel: Relative declination values at the three epochs (mas)
- ra_errs: Measurement errors for RA at the three epochs (mas)
- dec_errs: Measurement errors for Dec at the three epochs (mas)
"""
function simulate_astrometry(star::StarData, epoch::Real, dt::Real)
    # --- Time setup: generate observation epochs ---
    # Create a time array with three epochs: past, present, and future
    epochs_years = [epoch - dt, epoch, epoch + dt]
    # Convert the epochs from calendar years to Modified Julian Date (MJD)
    epochs_mjd = [Octofitter.years2mjd(y) for y in epochs_years]

    # --- Simulate astrometric positions at each epoch ---
    # Predict future and past RA/Dec based on position, proper motion, and acceleration
    future_ra = fake_pos(star.ra, star.pm_ra, star.acc_ra, dt)
    future_dec = fake_pos(star.dec, star.pm_dec, star.acc_dec, dt)
    past_ra = fake_pos(star.ra, star.pm_ra, star.acc_ra, -dt)
    past_dec = fake_pos(star.dec, star.pm_dec, star.acc_dec, -dt)

    # --- Propagate errors for simulated data points ---
    # Compute uncertainty in RA/Dec at past and future epochs using PM and acceleration errors
    future_ra_err = propagate_error(ustrip(ra_err), star.sigma_pm_ra, star.sigma_acc_ra, dt)
    future_dec_err = propagate_error(ustrip(dec_err), star.sigma_pm_dec, star.sigma_acc_dec, dt)
    past_ra_err = propagate_error(ustrip(ra_err), star.sigma_pm_ra, star.sigma_acc_ra, -dt)
    past_dec_err = propagate_error(ustrip(dec_err), star.sigma_pm_dec, star.sigma_acc_dec, -dt)

    # --- Relative positions ---
    # Combine past, present, and future positions into arrays
    ra_vals = [past_ra, star.ra, future_ra]
    dec_vals = [past_dec, star.dec, future_dec]
    # Subtract cluster center-of-mass (CM) position to get positions relative to CM
    ra_rel = ra_vals .- ra_cm_mas
    dec_rel = dec_vals .- dec_cm_mas
    # Set measurement errors (central epoch is assumed to be exact here with error ~1e-6)
    ra_errs = [past_ra_err, ustrip(ra_err), future_ra_err]
    dec_errs = [past_dec_err, ustrip(dec_err), future_dec_err]

    return epochs_mjd, ra_rel, dec_rel, ra_errs, dec_errs
end

# ========================================================
#  Angular to Linear Acceleration Conversion
# ========================================================
"""
Convert angular acceleration from milliarcseconds per year squared (mas/yr²)
to linear acceleration in kilometers per second squared (km/s²).

# Arguments
- a_masyr2::Quantity : Angular acceleration with units mas/yr²
- distance_km::Quantity : Distance to the object with units km

# Returns
- Linear acceleration in km/s² as a Quantity
"""
function masyr2_to_kms2(a_masyr2::Unitful.Quantity, distance_km::Unitful.Quantity)
    # Convert mas/yr² to rad/yr² using Unitful's dimensionless angles equivalency
    # 1 mas = 1e-3 arcsec = (1e-3 / 3600) deg = (1e-3 / 3600)*(π/180) rad
    # Unitful does not have built-in equivalencies, so we convert manually:
    a_radyr2 = uconvert(u"rad"/u"yr"^2, a_masyr2 * (1e-3 / 3600) * (π / 180))

    # Linear acceleration = angular acceleration × distance
    # a_radyr2 [rad/yr²] * distance_km [km] = km/yr²
    a_kmyr2 = a_radyr2 * distance_km

    # Convert km/yr² to km/s²
    a_kms2 = uconvert(u"km"/u"s"^2, a_kmyr2)

    return a_kms2
end

# ========================================================
#  Total Angular Accelerations and Uncertainty 
# ========================================================
"""
Compute total plane-of-sky angular and physical acceleration
(and their uncertainties) for a given star.

Parameters
----------
star : StarData
    A star object with acc_ra, acc_dec, sigma_acc_ra, and sigma_acc_dec attributes.

Returns
-------
a_total_masyr2 : Float64
    Total angular acceleration in mas/yr².
a_total_masyr2_err : Float64
    Uncertainty in total angular acceleration.
a_total_kms2 : Float64
    Total physical acceleration in km/s².
a_total_kms2_err : Float64
    Uncertainty in physical acceleration.
"""
function total_accelerations(star::StarData)
    a_ra = star.acc_ra
    a_dec = star.acc_dec
    a_ra_err = star.sigma_acc_ra
    a_dec_err = star.sigma_acc_dec

    # Angular acceleration magnitude
    a_total_masyr2 = sqrt(a_ra^2 + a_dec^2)

    # Uncertainty propagation
    a_total_masyr2_err = sqrt(
        (a_ra * a_ra_err / a_total_masyr2)^2 +
        (a_dec * a_dec_err / a_total_masyr2)^2
    )

    # Convert to physical acceleration [km/s²]
    a_total_kms2 = masyr2_to_kms2(a_total_masyr2 * u"mas/yr^2", distance_km)
    a_total_kms2_err = masyr2_to_kms2(a_total_masyr2_err * u"mas/yr^2", distance_km)

    return a_total_masyr2, a_total_masyr2_err, a_total_kms2, a_total_kms2_err
end

end