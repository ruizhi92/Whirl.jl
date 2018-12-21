mutable struct FSI{NX, NY, N}
    # Physical Parameters
    "Reynolds number"
    Re::Float64
    "Free stream velocities"
    U∞::Tuple{Float64, Float64}

    # Discretization
    "Grid spacing"
    Δx::Float64
    "Time step (used to determine integrating factor diffusion rate)"
    Δt::Float64
    "Runge-Kutta method"
    rk::RKParams

    # Operators
    "Laplacian operator"
    L::Fields.Laplacian{NX,NY}

    # Body coordinate data, if present
    # if a static problem, these coordinates are in inertial coordinates
    # if a non-static problem, in their own coordinate systems
    X̃::VectorData{N}

    # Pre-stored regularization and interpolation matrices (if present)
    Hmat::Nullable{RegularizationMatrix}
    Emat::Nullable{InterpolationMatrix}

    # Scratch space
    ## Pre-allocated space for intermediate values
    Vb::VectorData{N}
    Fq::Edges{Primal, NX, NY}
    Ww::Edges{Dual, NX, NY}
    Qq::Edges{Dual, NX, NY}

end

function FSI(dims::Tuple{Int, Int}, Re, Δx, Δt;
                       U∞ = (0.0, 0.0), X̃ = VectorData{0}(),
                       rk::RKParams=RK31)
    NX, NY = dims

    α = Δt/(Re*Δx^2)

    L = plan_laplacian((NX,NY),with_inverse=true)

    Vb = VectorData(X̃)
    Fq = Edges{Primal,NX,NY}()
    Ww   = Edges{Dual, NX, NY}()
    Qq  = Edges{Dual, NX, NY}()
    N = length(X̃)÷2

    # set interpolation and regularization matrices
    # Hmat = Nullable{RegularizationMatrix}()
    # Emat = Nullable{InterpolationMatrix}()
    regop = Regularize(X̃,Δx;issymmetric=true)
    Hmat, Emat = RegularizationMatrix(regop,VectorData{N}(),Edges{Primal,NX,NY}())

    FSI{NX, NY, N}(Re, U∞, Δx, Δt, rk, L, X̃, Hmat, Emat, Vb, Fq, Ww, Qq)
end

function Base.show(io::IO, sys::FSI{NX,NY,N}) where {NX,NY,N}
    print(io, "Fluid-Structure Interaction system on a grid of size $NX x $NY")
end

# Basic operators for any FSI system

# Integrating factor -- rescale the time-step size
Fields.plan_intfact(Δt,w,sys::FSI{NX,NY}) where {NX,NY} =
        Fields.plan_intfact(Δt/(sys.Re*sys.Δx^2),w)

# RHS of Navier-Stokes (non-linear convective term)
function TimeMarching.r₁(w::Nodes{Dual,NX,NY},t,sys::FSI{NX,NY}) where {NX,NY}

  Ww = sys.Ww
  Qq = sys.Qq
  L = sys.L
  Δx⁻¹ = 1/sys.Δx

  shift!(Qq,curl(L\w)) # -velocity, on dual edges
  Qq.u .-= sys.U∞[1]
  Qq.v .-= sys.U∞[2]

  return scale!(divergence(Qq∘shift!(Ww,w)),Δx⁻¹) # -∇⋅(wu)

end

# RHS of Navier-Stokes (non-linear convective term)
# U∞ represents free stream velocity, which is subtracted from the b.c.s from rigid body
function TimeMarching.r₁(w::Nodes{Dual,NX,NY},t,sys::FSI{NX,NY},U∞::RigidBodyMotions.RigidBodyMotion) where {NX,NY}

  Ww = sys.Ww
  Qq = sys.Qq
  L = sys.L
  Δx⁻¹ = 1/sys.Δx

  shift!(Qq,curl(L\w)) # -velocity, on dual edges
  _,ċ,_,_,_,_ = U∞(t)
  Qq.u .-= real(ċ)
  Qq.v .-= imag(ċ)

  return scale!(divergence(Qq∘shift!(Ww,w)),Δx⁻¹) # -∇⋅(wu)

end

# RHS of a stationary body with no surface velocity
function TimeMarching.r₂(w::Nodes{Dual,NX,NY},t,sys::FSI{NX,NY,N}) where {NX,NY,N}
    ΔV = VectorData(sys.X̃) # create a struct ΔV with same shape of sys.X̃ and initialize it
    ΔV.u .-= sys.U∞[1]
    ΔV.v .-= sys.U∞[2]
    return ΔV
end

function TimeMarching.r₂(w::Nodes{Dual,NX,NY},t,sys::FSI{NX,NY,N},U∞::RigidBodyMotions.RigidBodyMotion) where {NX,NY,N}
    ΔV = VectorData(sys.X̃) # create a struct ΔV with same shape of sys.X̃ and initialize it
    _,ċ,_,_,_,_ = U∞(t)
    ΔV.u .-= real(ċ)
    ΔV.v .-= imag(ċ)
    return ΔV
end

# Constraint operators, using stored regularization and interpolation operators
# B₁ᵀ = CᵀEᵀ, B₂ = -ECL⁻¹
TimeMarching.B₁ᵀ(f,sys::FSI{NX,NY,N}) where {NX,NY,N} = Curl()*(get(sys.Hmat)*f)
TimeMarching.B₂(w,sys::FSI{NX,NY,N}) where {NX,NY,N} = -(get(sys.Emat)*(Curl()*(sys.L\w)))

# Constructor using stored operators
TimeMarching.plan_constraints(w::Nodes{Dual,NX,NY},t,sys::FSI{NX,NY,N}) where {NX,NY,N} =
                    (f -> TimeMarching.B₁ᵀ(f,sys),w -> TimeMarching.B₂(w,sys))
