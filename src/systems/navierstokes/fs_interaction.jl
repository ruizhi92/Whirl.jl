# RHS of Navier-Stokes (non-linear convective term) for constant U∞
function TimeMarching.r₁(u::Tuple{Nodes{Dual,NX,NY},Vector{Float64}},t,sys::NavierStokes{NX,NY}) where {NX,NY}
    return TimeMarching.r₁(u[1],t,sys), zeros(Float64,length(u[2]))
end


# RHS of Navier-Stokes (non-linear convective term) for time-varying U∞(t)
# U∞ represents free stream velocity, which is subtracted from the b.c.s from rigid body
function TimeMarching.r₁(u::Tuple{Nodes{Dual,NX,NY},Vector{Float64}},t,sys::NavierStokes{NX,NY},
                            U∞::RigidBodyMotions.RigidBodyMotion) where {NX,NY}
    return TimeMarching.r₁(u[1],t,sys,U∞), zeros(Float64,length(u[2]))
end

"""
Boundary condition for uniform constrant free stream U∞
r₂ takes in motion as linear velocity of body points in the inertial frame and substract
the free stream velocity.
"""
function TimeMarching.r₂(u::Tuple{Nodes{Dual,NX,NY},Vector{Float64}},t,sys::NavierStokes{NX,NY,N,false},
                            ) where {NX,NY,N}

    motion = u[2][end-round(Int,length(u[2])/2)+1:end]
    ΔV = VectorData(reshape(motion,:,2))
    ΔV.u .-= sys.U∞[1]
    ΔV.v .-= sys.U∞[2]
# println("motion ", ΔV.v, "\n")
    return ΔV, Vector{Float64}()

end

"""
Boundary condition for uniform time-varying free stream U∞(t)
r₂ takes in motion as linear velocity of body points in the inertial frame and substract
the free stream velocity.
"""
function TimeMarching.r₂(u::Tuple{Nodes{Dual,NX,NY},Vector{Float64}},t,sys::NavierStokes{NX,NY,N,true},
                U∞::RigidBodyMotions.RigidBodyMotion) where {NX,NY,N}

    motion = u[2][end-round(Int,length(u[2])/2)+1:end]
    ΔV = VectorData(reshape(motion,:,2))
    _,ċ,_,_,_,_ = U∞(t)
    ΔV.u .-= real(ċ)
    ΔV.v .-= imag(ċ)
    return ΔV, Vector{Float64}()
end

# Constraint operator constructors
"""
plan_constraints takes in coordinates of body points in inertial frame and construct
B₁ᵀ, B₂ operator from it.
"""
function TimeMarching.plan_constraints(u::Tuple{Nodes{Dual,NX,NY},Vector{Float64}},
                            t,sys::NavierStokes{NX,NY,N,false}) where {NX,NY,N}

    coord = u[2][1:round(Int,length(u[2])/2)]
# println("coord ", coord,"\n")

    X = VectorData(reshape(coord,:,2))
    regop = Regularize(X,sys.Δx;issymmetric=true)
    if sys._isstore
      Hmat, Emat = RegularizationMatrix(regop,VectorData{N}(),Edges{Primal,NX,NY}())
      sys.Hmat = Hmat
      sys.Emat = Emat
      return (f->TimeMarching.B₁ᵀ(f,sys), f->zeros(Float64,size(u[2]))),
             (w->TimeMarching.B₂(w,sys), u->Vector{Float64}())
    else
      return (f->TimeMarching.B₁ᵀ(f,regop,sys), f->zeros(Float64,size(u[2]))),
             (w->TimeMarching.B₂(w,regop,sys), u->Vector{Float64}())
    end
end
