"""
    AbstractApproximationMethod

The supertype of the approximation methods: [`Unscented`](@ref), [`Linearization`](@ref) and
[`GaussHermiteCubature`](@ref). A method implements what its callers need: [`approximate`](@ref)
to propagate moments through a function, or [`getpoints`](@ref) and [`getweights`](@ref) for a
cubature, which gives it [`approximate_meancov`](@ref); and [`approximation_name`](@ref) and
[`approximation_short_name`](@ref) for display.
"""
abstract type AbstractApproximationMethod end

"""
    approximation_name(method::AbstractApproximationMethod)

The method's full name, for display: `"Unscented"`, `"Linearization"`, or
`"GaussHermite(p)"` with its number of points. See also [`approximation_short_name`](@ref).
"""
function approximation_name end

"""
    approximation_short_name(method::AbstractApproximationMethod)

The method's abbreviated name, for display: `"UT"`, `"LN"`, or `"GHp"` with its number of
points. See also [`approximation_name`](@ref).
"""
function approximation_short_name end
