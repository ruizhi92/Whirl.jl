module TimeMarching

import Whirl:@get, Nodes
import Dyn3d:RKParams, TimeMarching.RK31

export System, Constrained, Unconstrained, RK, IFRK, IFHERK, r₁, r₂, B₂, B₁ᵀ,
          plan_constraints, RKParams, RK31

"Abstract type for a system of ODEs"
abstract type System{C} end

const Constrained = true
const Unconstrained = false

# Functions that get extended by individual systems
function r₁ end
function r₂ end
function B₂ end
function B₁ᵀ end
function plan_constraints end

using ..SaddlePointSystems

# include("timemarching/rk.jl")
# include("timemarching/ifrk.jl")
include("timemarching/ifherk.jl")
# include("timemarching/ifherk_fc2d.jl")

end
