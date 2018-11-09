# RHS of Navier-Stokes (non-linear convective term) for constant U∞
"""
Boundary condition at body points for fluids with uniform constrant free stream U∞
r₂ takes in motion as linear velocity of body points in the inertial frame and substract
the free stream velocity.
"""
function TimeMarching.r₂(u::Nodes{Dual,NX,NY},t,sys::NavierStokes{NX,NY,N,false},
            motion::Array{Float64,2}) where {NX,NY,N}
    ΔV = VectorData(motion)
    ΔV.u .-= sys.U∞[1]
    ΔV.v .-= sys.U∞[2]
# println("motion ", ΔV.v, "\n")
    return ΔV
end

"""
Boundary condition at body points for fluids with uniform time-varying free stream U∞(t)
r₂ takes in motion as linear velocity of body points in the inertial frame and substract
the free stream velocity.
"""
function TimeMarching.r₂(u::Nodes{Dual,NX,NY},t,sys::NavierStokes{NX,NY,N,false},
            motion::Array{Float64,2},U∞::RigidBodyMotions.RigidBodyMotion) where {NX,NY,N}
    ΔV = VectorData(motion)
    _,ċ,_,_,_,_ = U∞(t)
    ΔV.u .-= real(ċ)
    ΔV.v .-= imag(ċ)
    return ΔV
end

"""
plan_constraints takes in coordinates of body points in inertial frame and construct
B₁ᵀ, B₂ operator from it.
"""
function TimeMarching.plan_constraints(u::Nodes{Dual,NX,NY},t,sys::NavierStokes{NX,NY,N,false},
            coord::Array{Float64,2}) where {NX,NY,N}
# println("coord ", coord,"\n")
    X = VectorData(coord)
    regop = Regularize(X,sys.Δx;issymmetric=true)
    if sys._isstore
      Hmat, Emat = RegularizationMatrix(regop,VectorData{N}(),Edges{Primal,NX,NY}())
      sys.Hmat = Hmat
      sys.Emat = Emat
      return f->TimeMarching.B₁ᵀ(f,sys), w->TimeMarching.B₂(w,sys)
    else
      return f->TimeMarching.B₁ᵀ(f,regop,sys), w->TimeMarching.B₂(w,regop,sys)
    end
end
