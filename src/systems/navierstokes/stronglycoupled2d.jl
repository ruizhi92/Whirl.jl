"""
r₂ takes in linear velocity of body points in the inertial frame and substract
the free stream velocity.
"""
function TimeMarching.r₂(u::Nodes{Dual,NX,NY},t,sys::NavierStokes{NX,NY,N,false},
                            motion::Array{Float64,2}) where {NX,NY,N}

  ΔV = VectorData(motion)
  ΔV.u .-= sys.U∞[1]
  ΔV.v .-= sys.U∞[2]
  return ΔV, Vector{Float64}()
  
end

"""
plan_constraints takes in coordinates of body points in inertial frame and construct
B₁ᵀ, B₂ operator from it.
"""
function TimeMarching.plan_constraints(u::Nodes{Dual,NX,NY},t,sys::NavierStokes{NX,NY,N,false}
                            coord::Array{Float64,2}) where {NX,NY,N}

  X = VectorData(coord)
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
